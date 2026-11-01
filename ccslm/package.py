"""Build the `mt_slm_ctx_compress` repository from a model folder and run the validators' checks on it.

The repository is what the miner template's `slm_repo/` describes: weights, config, tokenizer files
(byte identical to the pinned base), a model card under the Microtensor submission licence, LICENSE,
and the training record under `training/` (recipe with the declared base, data sources, the run's
config, metrics, env and final eval, and the compression record when the model is pruned or
quantised). Checks, all from microtensor_subnet: the preflight that `mt check-model` runs (config,
tokenizer, headers, parameter bounds, footprint at the arena's context length within 24 GB),
`check_slm_repo` (required files, training record, allowed base), the card licence, the weight
fingerprint against the base (>= 0.9) and that no weight file is byte identical to the base.

  python -m ccslm.package --model out/runs/r1/merged --run out/runs/r1 --eval out/eval/r1/report.json \
      --out out/slm_repo --hf-repo <user>/mt_slm_ctx_compress
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ccslm.common import BASE, BASE_REVISION, ROOT, log, read_json, write_json, write_portable_config

MAX_LEN = 122_880 + 3_072
MEMORY_BYTES = 25_769_803_776
TEMPLATE_LICENSE = ROOT.parent / "microtensor-miner-template" / "slm_repo" / "LICENSE"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty", "frequency_penalty", "do_sample")
CARD = """---
license: other
license_name: microtensor-submission
license_link: https://www.microtensor.cloud/terms
base_model: {base}
tags:
- microtensor
- ctx_compress
---

# {name}

A specialised model for the Microtensor `ctx_compress` arena (agent session compaction): it reads a
coding agent session of up to 120k tokens and writes the nine heading compaction summary.

| Field | Value |
|---|---|
| Base model | `{base}` at `{revision}` (Apache 2.0) |
| Method | {method} |
| Arena | ctx_compress |
| Training data | sessions from the arena's own generator (`microtensor-io/microtensor-arenas`, `arenas/compress/synth.py`) under private seeds |
| Held out result | {result} |

The root loads with `AutoModelForCausalLM`. Everything under `training/` is the record validators
require and the archive keeps; it is never scored.
"""


def computed_memory(model_dir: Path, max_len: int = MAX_LEN) -> dict[str, Any]:
    """The arena's computed memory: parameters x bytes per weight + the KV cache formula."""
    from microtensor_subnet.artifact.params import kv_cache_bytes, model_parameters
    from microtensor_subnet.validator.transcripts import computed_memory as formula
    from microtensor_subnet.validator.transcripts import weight_format

    from ccslm.common import definition

    parameters = model_parameters(model_dir)
    kind = weight_format(model_dir)
    kv = kv_cache_bytes(model_dir, max_len)
    try:
        found = definition()
    except Exception:  # noqa: BLE001 - offline: the table defaults are the same numbers
        found = {}
    return {"parameters": parameters, "format": kind, "kv_bytes": kv, "memory_bytes": formula(parameters, kind, kv, found)}


def build(model_dir: Path, out: Path, *, run_dir: Path | None, eval_report: Path | None, extra: dict[str, Any]) -> None:
    from ccslm.common import base_dir

    base = base_dir()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for path in sorted(model_dir.glob("*.safetensors")):
        shutil.copyfile(path, out / path.name)
    for name in ("model.safetensors.index.json", "config.json", "generation_config.json"):
        if (model_dir / name).is_file():
            shutil.copyfile(model_dir / name, out / name)
    write_portable_config(out, base)
    generation = read_json(out / "generation_config.json", {}) or {}
    cleaned = {k: v for k, v in generation.items() if k not in SAMPLING_KEYS}
    if cleaned != generation:
        write_json(out / "generation_config.json", cleaned)
    for name in TOKENIZER_FILES:
        if (base / name).is_file():
            shutil.copyfile(base / name, out / name)
    if TEMPLATE_LICENSE.is_file():
        shutil.copyfile(TEMPLATE_LICENSE, out / "LICENSE")
    else:
        (out / "LICENSE").write_text(
            "Licensed under the Microtensor submission terms: https://www.microtensor.cloud/terms\n",
            encoding="utf-8",
        )
    run_config = read_json(run_dir / "config.json", {}) if run_dir else {}
    final = read_json(eval_report, {}) if eval_report else {}
    method = run_config.get("method", "fine tune")
    if run_config.get("rank"):
        method = f"LoRA r={run_config['rank']} merged into bf16"
    if extra.get("pruning"):
        method += f", {len(extra['pruning'].get('dropped', []))} Mamba layers pruned"
    if extra.get("quantisation"):
        method += ", GPTQ 4 bit"
    result = (
        f"pass rate {final.get('pass_rate')} (Wilson LB {final.get('lower_bound')}) on {final.get('n')} hidden-style items"
        if final
        else "see training/runs"
    )
    name = extra.get("name") or "mt_slm_ctx_compress"
    (out / "README.md").write_text(
        CARD.format(base=BASE, revision=BASE_REVISION[:12], name=name, method=method, result=result),
        encoding="utf-8",
    )
    training = out / "training"
    recipe = {
        "base_model": BASE,
        "base_revision": BASE_REVISION,
        "method": "supervised fine tuning on generated sessions with grader-checked targets",
        "harness_aware": False,
        "hyperparameters": {
            k: run_config.get(k)
            for k in ("method", "rank", "lora_targets", "learning_rate", "epochs", "accumulate", "max_tokens")
            if k in run_config
        },
        "seeds": [run_config.get("seed", 0)],
        "pruning": extra.get("pruning") or None,
        "compression": "gptq-int4" if extra.get("quantisation") else "none",
    }
    write_json(training / "recipe.json", recipe)
    write_json(
        training / "data" / "sources.json",
        [
            {
                "name": "microtensor-io/microtensor-arenas session generator (arenas/compress/synth.py)",
                "version": extra.get("arenas_commit", ""),
                "license": "Microtensor dataset terms",
                "sha256": "",
            },
            {"name": "own-synthetic targets (ccslm/targets.py)", "version": "1", "license": "proprietary", "sha256": ""},
        ],
    )
    (training / "data" / "data_card.md").write_text(
        "Sessions come from the arena's own generator under private seeds (train, val and test disjoint).\n"
        "Targets are nine heading summaries written from each session up to its cut and checked with the\n"
        "arena grader's recall, hallucination, heading and budget rules before use.\n",
        encoding="utf-8",
    )
    run_id = run_dir.name if run_dir else "run"
    runs = training / "runs" / run_id
    runs.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "metrics.jsonl", "env.json"):
        if run_dir and (run_dir / name).is_file():
            shutil.copyfile(run_dir / name, runs / name)
    if not (runs / "metrics.jsonl").is_file():
        (runs / "metrics.jsonl").write_text("", encoding="utf-8")
    write_json(runs / "final_eval.json", final or {"note": "fill from ccslm.grade on the test split"})
    if extra.get("pruning") or extra.get("quantisation"):
        write_json(training / "compression" / "compression.json", {"pruning": extra.get("pruning"), "quantisation": extra.get("quantisation")})
        if extra.get("before_after"):
            write_json(training / "compression" / "before_after.json", extra["before_after"])
    scripts = training / "scripts" / "ccslm"
    scripts.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "ccslm").glob("*.py")):
        shutil.copyfile(source, scripts / source.name)


def checks(root: Path, bounds: tuple[int, int] = (100_000_000, 8_000_000_000)) -> dict[str, Any]:
    from microtensor_subnet.artifact.validate import check_slm_repo, sha256_of
    from microtensor_subnet.validator import preflight
    from microtensor_subnet.validator.collect import card_value
    from microtensor_subnet.validator.provenance import fingerprint_similarity

    from ccslm.common import base_dir

    base = base_dir()
    problems: list[str] = []
    found = preflight.check(root, profile="P1", bounds=bounds, max_len=MAX_LEN, memory_bytes=MEMORY_BYTES)
    result: dict[str, Any] = {
        "preflight": {
            "message": found.message,
            "parameters": found.parameters,
            "footprint_bytes": found.footprint_bytes,
            "weights_sha256": found.weights_sha256,
        }
    }
    if not found.ok:
        problems.append(f"preflight {found.message}")
    repo = check_slm_repo(root, None, [BASE], bounds=bounds, formats=["bf16", "fp16", "awq", "gptq"])
    problems += repo.problems
    licence = card_value(root, "license")
    if not (licence == "other" and card_value(root, "license_name") == "microtensor-submission") and licence != "apache-2.0":
        problems.append(f"card licence {licence!r} is not accepted")
    similarity = fingerprint_similarity(root, base)
    result["fingerprint_similarity"] = round(similarity, 6)
    if similarity < 0.9:
        problems.append(f"fingerprint similarity {similarity:.4f} is under 0.9")
    base_hashes = {sha256_of(p) for p in base.glob("*.safetensors")}
    same = [p.name for p in root.glob("*.safetensors") if sha256_of(p) in base_hashes]
    if same:
        problems.append(f"weight files identical to the base: {same}")
    for name in TOKENIZER_FILES:
        if (base / name).is_file() and (root / name).read_bytes() != (base / name).read_bytes():
            problems.append(f"{name} differs from the base")
    config = read_json(root / "config.json", {})
    if config.get("architectures") != ["GraniteMoeHybridForCausalLM"]:
        problems.append(f"architectures is {config.get('architectures')}")
    reference = read_json(base / "config.json", {})
    moved = sorted(
        k for k in set(config) | set(reference)
        if k not in ("num_hidden_layers", "layer_types", "quantization_config") and config.get(k) != reference.get(k)
    )
    if moved:
        problems.append(f"config.json differs from the base in {moved}")
    if any(t not in ("mamba", "attention") for t in config.get("layer_types") or []):
        problems.append("layer_types are not in the base's mamba/attention naming")
    generation = read_json(root / "generation_config.json", {}) or {}
    if any(k in generation for k in SAMPLING_KEYS):
        problems.append("generation_config.json sets sampling defaults that vLLM would apply")
    result["efficiency_inputs"] = computed_memory(root)
    result["efficiency_inputs_base"] = computed_memory(base)
    result["problems"] = problems
    result["ok"] = not problems
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="build and check the slm repository")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", default="", help="the training run folder (config, metrics, env)")
    parser.add_argument("--eval", default="", help="grade report.json on the test split")
    parser.add_argument("--prune-report", default="")
    parser.add_argument("--quant-report", default="")
    parser.add_argument("--hf-repo", default="", help="for the card title only; nothing is uploaded")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if not args.check_only:
        extra = {
            "name": args.hf_repo.split("/")[-1] if args.hf_repo else "",
            "pruning": read_json(args.prune_report) if args.prune_report else None,
            "quantisation": read_json(args.quant_report) if args.quant_report else None,
        }
        build(Path(args.model), out, run_dir=Path(args.run) if args.run else None, eval_report=Path(args.eval) if args.eval else None, extra=extra)
    result = checks(out)
    write_json(out.parent / f"{out.name}.checks.json", result)
    log(json.dumps({"ok": result["ok"], "problems": result["problems"], "fingerprint": result["fingerprint_similarity"]}))


if __name__ == "__main__":
    main()
