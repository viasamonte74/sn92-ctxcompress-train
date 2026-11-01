"""One command for the whole run: each stage resumable, logged to out/logs, marked done in out/stages.

  python -m ccslm.pipeline --stages check,data,targets            # CPU box: no GPU needed
  python -m ccslm.pipeline --stages baseline,train,eval,speed,package --variant A   # GPU box
  python -m ccslm.pipeline --stages report
  python -m ccslm.pipeline --stages export,eval --variant A --step 100   # one checkpoint, as A-s00100

Stages
  check     re-read the live arena definition and stop if the task moved under the project
  data      build test, val and train items with the arena's generator (CPU, ~1 h for 1,000 sessions)
  targets   write {prompt, completion} pairs for train and val; reject any target the grader would fault
  oracle    grade the test split's own targets with the probe (the ceiling of this target format)
  baseline  serve the untrained base with vLLM, generate on test, grade: the score to beat
  prune     variants C and D: score layers on val, cut, write out/prune/<variant>
  train     fine tune the variant (LoRA by default) and merge
  quant     variants B and D: GPTQ 4 bit of the merged model (experimental)
  eval      serve the variant, generate on test, grade, paired lower bound against the base
  speed     concurrency 1 timings of the variant and the base on the longest test items, envelope gate
  package   slm_repo for the variant and the validators' checks
  export    merge a numbered training checkpoint (--step N) into out/runs/<variant>/merged-step-N
  report    the variants table: pass rate, LB, twin factor, paired LB, E, estimated SLM score
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ccslm.common import ROOT, Stages, check_definition, definition, log, out_dir, read_json, write_json

VARIANTS: dict[str, dict[str, Any]] = {
    # A: LoRA on the base. B: A in GPTQ 4 bit.
    "A": {"start": "base", "method": "lora", "rank": 64, "quant": False, "drop": 0},
    "B": {"from": "A", "quant": True},
    # C: the fewest Mamba layers that bring the model under 3.0B parameters (3 for Granite H Micro),
    # which halves the per token price in the system and escalation pools; then LoRA. D: C in GPTQ.
    "C": {"start": "pruned", "method": "lora", "rank": 64, "quant": False, "max_params": 3_000_000_000},
    "D": {"from": "C", "quant": True},
    # F: a full fine tune of the base (fp32 masters, 8 bit moments). G: 8 layers cut, full fine tune.
    "F": {"start": "base", "method": "full", "rank": 0, "quant": False, "drop": 0},
    "G": {"start": "pruned", "method": "full", "rank": 0, "quant": False, "drop": 8},
}
SPLIT_SESSIONS = {"test": 100, "val": 60, "train": 1000}
SPEED_ITEMS = 30
PORT = int(os.environ.get("CCSLM_VLLM_PORT", "8000"))
# --step N: export, eval, speed and package work on training checkpoint N instead of the final merge.
STEP = 0


def tag(variant: str) -> str:
    return f"{variant}-s{STEP:05d}" if STEP else variant


def shell(args: list[str], log_name: str) -> None:
    logs = out_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log(" ".join(args))
    with (logs / f"{log_name}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(args)}\n")
        handle.flush()
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stderr.write(line)
            handle.write(line)
        if process.wait() != 0:
            raise SystemExit(f"{args[2] if len(args) > 2 else args[0]} failed; see out/logs/{log_name}.log")


def python(module: str, *args: str, log_name: str) -> None:
    shell([sys.executable, "-m", module, *args], log_name)


@contextmanager
def served(model: Path, name: str, extra: list[str] | None = None) -> Iterator[str]:
    logs = out_dir() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    handle = (logs / f"vllm-{name}.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts" / "serve_vllm.sh"), str(model), str(PORT), *(extra or [])],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{PORT}"
    try:
        deadline = time.time() + 1200
        while time.time() < deadline:
            if process.poll() is not None:
                raise SystemExit(f"vLLM exited; see out/logs/vllm-{name}.log")
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=5):  # noqa: S310
                    break
            except OSError:
                time.sleep(5)
        else:
            raise SystemExit("vLLM did not come up in 20 minutes")
        log(f"vLLM serving {model} at {url}")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        handle.close()


def probe_args() -> list[str]:
    url = os.environ.get("CCSLM_PROBE_URL", "")
    return ["--probe", "openai", "--probe-url", url] if url else ["--probe", "hf"]


def resolve(variant: str) -> dict[str, Any]:
    spec = dict(VARIANTS[variant])
    if "from" in spec:
        parent = resolve(spec["from"])
        parent.update({k: v for k, v in spec.items() if k != "from"})
        parent["parent"] = spec["from"]
        return parent
    return spec


def model_of(variant: str) -> Path:
    spec = resolve(variant)
    out = out_dir()
    if spec.get("quant"):
        return out / "quant" / tag(variant)
    if STEP:
        return out / "runs" / variant / f"merged-step-{STEP:05d}"
    return out / "runs" / variant / "merged"


def stage_check(_: str) -> None:
    found = definition(refresh=True)
    problems = check_definition(found)
    timing = read_json(out_dir() / "arena.json", {}).get("timing")
    log(f"arena timing {json.dumps(timing)}")
    if problems:
        raise SystemExit("the live arena moved: " + "; ".join(problems))
    log("the live definition matches what this project was built for")


def split_sessions() -> dict[str, int]:
    found = dict(SPLIT_SESSIONS)
    for part in os.environ.get("CCSLM_SPLIT_SESSIONS", "").split(","):
        if "=" in part:
            name, _, value = part.partition("=")
            found[name.strip()] = int(value)
    return found


def stage_data(_: str) -> None:
    for split, sessions in split_sessions().items():
        target = out_dir() / "data" / split
        if (target / "items.jsonl").is_file():
            log(f"{split} items exist, skipping")
            continue
        python("ccslm.data", "--split", split, "--sessions", str(sessions), "--out", str(target), log_name="data")


def stage_targets(_: str) -> None:
    for split in ("train", "val", "test"):
        folder = out_dir() / "data" / split
        python("ccslm.targets", "--items", str(folder / "items.jsonl"), "--out", str(folder / "pairs.jsonl"), log_name="targets")


def stage_oracle(_: str) -> None:
    folder = out_dir() / "data" / "test"
    python("ccslm.grade", "--items", str(folder / "items.jsonl"), "--oracle", *probe_args(), "--out", str(out_dir() / "eval" / "oracle"), log_name="oracle")


def generate_and_grade(model: Path, name: str, compare: str = "", extra: list[str] | None = None) -> None:
    out = out_dir()
    gen = out / "gen" / f"{name}.jsonl"
    with served(model, name, extra) as url:
        python("ccslm.generate", "--items", str(out / "data" / "test" / "items.jsonl"), "--out", str(gen), "--server", url, "--concurrency", "8", log_name=f"eval-{name}")
    args = ["--items", str(out / "data" / "test" / "items.jsonl"), "--outputs", str(gen), *probe_args(), "--out", str(out / "eval" / name)]
    if compare:
        args += ["--compare", str(out / "eval" / compare / "graded.jsonl")]
    python("ccslm.grade", *args, log_name=f"eval-{name}")


def stage_baseline(_: str) -> None:
    from ccslm.common import base_dir

    generate_and_grade(base_dir(), "base")


def stage_prune(variant: str) -> None:
    spec = resolve(variant)
    if not spec.get("drop") and not spec.get("max_params"):
        log(f"variant {variant} is not pruned")
        return
    from ccslm.common import base_dir

    out = out_dir() / "prune"
    scores = out / "scores.json"
    if not scores.is_file():
        python("ccslm.prune", "score", "--model", str(base_dir()), "--pairs", str(out_dir() / "data" / "val" / "pairs.jsonl"), "--samples", "32", "--out", str(out), log_name="prune")
    how = ["--max-params", str(spec["max_params"])] if spec.get("max_params") else ["--drop", str(spec["drop"])]
    python("ccslm.prune", "cut", "--model", str(base_dir()), "--scores", str(scores), *how, "--out", str(out / variant), log_name="prune")


def stage_train(variant: str) -> None:
    spec = resolve(variant)
    if spec.get("parent"):
        log(f"variant {variant} reuses the training of {spec['parent']}")
        return
    data = out_dir() / "data"
    args = ["--pairs", str(data / "train" / "pairs.jsonl"), "--eval", str(data / "val" / "pairs.jsonl"), "--out", str(out_dir() / "runs" / variant), "--method", spec["method"]]
    if spec["method"] == "lora":
        args += ["--rank", str(spec["rank"])]
    if spec["start"] == "pruned":
        args += ["--model", str(out_dir() / "prune" / variant)]
    if (out_dir() / "runs" / variant / "checkpoint").is_dir():
        args.append("--resume")
    args += os.environ.get("CCSLM_TRAIN_ARGS", "").split()
    python("ccslm.train", *args, log_name=f"train-{variant}")


def stage_quant(variant: str) -> None:
    spec = resolve(variant)
    if not spec.get("quant"):
        log(f"variant {variant} is not quantised")
        return
    source = out_dir() / "runs" / spec["parent"] / (f"merged-step-{STEP:05d}" if STEP else "merged")
    python("ccslm.quantize", "run", "--model", str(source), "--pairs", str(out_dir() / "data" / "train" / "pairs.jsonl"), "--out", str(model_of(variant)), log_name=f"quant-{variant}")


def stage_eval(variant: str) -> None:
    extra = ["--quantization", "gptq_marlin"] if resolve(variant).get("quant") else []
    generate_and_grade(model_of(variant), tag(variant), compare="base", extra=extra)


def stage_speed(variant: str) -> None:
    from ccslm.common import base_dir

    out = out_dir()
    items = str(out / "data" / "test" / "items.jsonl")
    for name, model, extra in (("base", base_dir(), []), (tag(variant), model_of(variant), ["--quantization", "gptq_marlin"] if resolve(variant).get("quant") else [])):
        gen = out / "speed" / f"{name}.jsonl"
        if name == "base" and gen.is_file():
            continue
        with served(model, f"speed-{name}", extra) as url:
            python("ccslm.generate", "--items", items, "--out", str(gen), "--server", url, "--concurrency", "1", "--limit", str(SPEED_ITEMS), "--longest-first", log_name=f"speed-{name}")
        python("ccslm.generate", "envelope", "--outputs", str(gen), "--out", str(out / "speed" / f"{name}.envelope.json"), log_name=f"speed-{name}")


def stage_package(variant: str) -> None:
    out = out_dir()
    spec = resolve(variant)
    run = out / "runs" / (spec.get("parent") or variant)
    name = tag(variant)
    args = ["--model", str(model_of(variant)), "--out", str(out / "slm_repo" / name), "--run", str(run)]
    if (out / "eval" / name / "report.json").is_file():
        args += ["--eval", str(out / "eval" / name / "report.json")]
    if spec.get("drop") or spec.get("max_params"):
        args += ["--prune-report", str(out / "prune" / (spec.get("parent") or variant) / "prune_report.json")]
    if spec.get("quant"):
        args += ["--quant-report", str(model_of(variant) / "quant_report.json")]
    python("ccslm.package", *args, log_name=f"package-{variant}")


def stage_export(variant: str) -> None:
    spec = resolve(variant)
    run = out_dir() / "runs" / (spec.get("parent") or variant)
    if not STEP:
        raise SystemExit("the export stage needs --step N (see python -m ccslm.export --run %s --list)" % run)
    python("ccslm.export", "--run", str(run), "--step", str(STEP), "--out", str(out_dir() / "runs" / (spec.get("parent") or variant) / f"merged-step-{STEP:05d}"), log_name=f"export-{variant}")


def stage_report(_: str) -> None:
    from ccslm.grade import estimate

    out = out_dir()
    base_speed = read_json(out / "speed" / "base.envelope.json", {})
    base_checks = None
    rows = []
    evaluated = sorted(p.name for p in (out / "eval").glob("*") if (p / "report.json").is_file() and p.name not in ("oracle", "base"))
    for variant in [*evaluated, "base"]:
        report = read_json(out / "eval" / variant / "report.json")
        if not report:
            continue
        speed = read_json(out / "speed" / f"{variant}.envelope.json", {})
        checks = read_json(out / "slm_repo" / f"{variant}.checks.json", {})
        if checks and base_checks is None:
            base_checks = checks.get("efficiency_inputs_base")
        mine = None
        if speed and checks:
            mine = {"ttft": speed.get("ttft_p95_ms"), "tpot": speed.get("tpot_p95_ms"), "memory": checks["efficiency_inputs"]["memory_bytes"], "parameters": checks["efficiency_inputs"]["parameters"]}
        reference = None
        if base_speed and base_checks:
            reference = {"ttft": base_speed.get("ttft_p95_ms"), "tpot": base_speed.get("tpot_p95_ms"), "memory": base_checks["memory_bytes"], "parameters": base_checks["parameters"]}
        from ccslm.estimate import costs

        parameters = (checks.get("efficiency_inputs") or {}).get("parameters") if checks else None
        priced = costs(parameters, 60_000, int(report.get("output_tokens_mean") or 720)) if parameters else {}
        rows.append(
            {
                "variant": variant,
                "parameters": parameters,
                "price_per_token_micros": priced.get("price_per_token_micros"),
                "cheaper_than_haiku": priced.get("vs_claude-haiku-5-5"),
                "cheaper_than_glm_flash": priced.get("vs_glm-5-3-flash"),
                "pass_rate": report["pass_rate"],
                "lower_bound": report["lower_bound"],
                "twin_factor": report["twin_factor"],
                "paired_lb_vs_base": (report.get("paired") or {}).get("lower_bound"),
                "envelope_pass": speed.get("envelope_pass"),
                "checks_ok": checks.get("ok"),
                **estimate(report, mine, reference),
            }
        )
    write_json(out / "variants.json", rows)
    header = f"{'variant':8} {'pass':>6} {'LB':>6} {'twin':>5} {'pairedLB':>8} {'E':>6} {'S_slm':>6} {'price':>6} {'xHaiku':>7} envelope checks"
    print(header)
    for r in rows:
        print(
            f"{r['variant']:8} {r['pass_rate']:6.3f} {r['lower_bound']:6.3f} {r['twin_factor']:5.2f} "
            f"{(r['paired_lb_vs_base'] if r['paired_lb_vs_base'] is not None else float('nan')):8.3f} "
            f"{r['efficiency']:6.3f} {r['slm_score']:6.3f} {str(r['price_per_token_micros']):>6} "
            f"{str(r['cheaper_than_haiku']):>7} {r['envelope_pass']} {r['checks_ok']}"
        )


STAGES = {
    "check": stage_check,
    "data": stage_data,
    "targets": stage_targets,
    "oracle": stage_oracle,
    "baseline": stage_baseline,
    "prune": stage_prune,
    "train": stage_train,
    "quant": stage_quant,
    "eval": stage_eval,
    "speed": stage_speed,
    "package": stage_package,
    "export": stage_export,
    "report": stage_report,
}
PER_VARIANT = {"prune", "train", "quant", "eval", "speed", "package", "export"}
PER_STEP = {"export", "quant", "eval", "speed", "package"}


def main() -> None:
    parser = argparse.ArgumentParser(description="run the ctx_compress model pipeline")
    parser.add_argument("--stages", required=True, help=f"comma separated, from {','.join(STAGES)}")
    parser.add_argument("--variant", default="A", choices=sorted(VARIANTS))
    parser.add_argument("--step", type=int, default=0, help="training checkpoint for export/quant/eval/speed/package (default: the final merge)")
    parser.add_argument("--force", action="store_true", help="re-run stages already marked done")
    args = parser.parse_args()
    global STEP
    STEP = args.step
    stages = Stages()
    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}")
        marker = f"{name}-{tag(args.variant) if name in PER_STEP else args.variant}" if name in PER_VARIANT else name
        if stages.done(marker) and not args.force and name not in ("check", "report"):
            log(f"stage {marker} already done")
            continue
        log(f"stage {marker} start")
        started = time.time()
        STAGES[name](args.variant)
        stages.mark(marker, {"seconds": round(time.time() - started, 1)})
        log(f"stage {marker} done in {time.time() - started:.0f}s")
    missing = [p for p in (out_dir() / "arena.json",) if not p.is_file()]
    if missing:
        log(f"note: {missing[0]} is missing; run the check stage")


if __name__ == "__main__":
    main()
