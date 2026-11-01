"""4 bit GPTQ of a merged model, and the format checks the validators apply to quantised weights.

What the validator code accepts (artifact/params.py, validator/provenance.py, the mt-5 envelope):
safetensors with the legacy GPTQ or AWQ layout (`*.qweight`, `*.qzeros`, `*.scales`, plus `*.g_idx`
for GPTQ; int32 packed, 8 values per element), `config.json` `quantization_config.quant_method`
"gptq" or "awq", and nothing below 4 bit. Other layouts (compressed-tensors `weight_packed`,
bitsandbytes) are invisible to the fingerprint and miscounted by the parameter counter, so they are
refused here.

The gain is the computed memory (0.5 instead of 2 bytes per weight) and some decode speed; the
parameter count does not drop (a packed int32 counts as 8 weights) and long prefill can get slower,
which costs e_TTFT, the largest efficiency weight. Measure before adopting it.

Experimental: GPTQModel must know the granitemoehybrid architecture and vLLM must serve GPTQ for it.
Both are checked when this runs; nothing here was run on a GPU yet.

  python -m ccslm.quantize run --model out/runs/r1/merged --pairs out/data/train/pairs.jsonl --out out/quant/r1-gptq
  python -m ccslm.quantize check --model out/quant/r1-gptq
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ccslm.common import iter_jsonl, log, write_json, write_portable_config

BITS = 4
GROUP = 128
PACKED = ("I32", "U32")
FORBIDDEN_SUFFIXES = (".weight_packed", ".weight_scale", ".absmax", ".quant_map", ".quant_state")


def check_format(model_dir: Path) -> list[str]:
    from microtensor_subnet.artifact.params import read_header, safetensors_in

    problems: list[str] = []
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["config.json is missing or unreadable"]
    quant = config.get("quantization_config") or {}
    method = str(quant.get("quant_method") or "").lower()
    if method not in ("gptq", "awq"):
        problems.append(f"quantization_config.quant_method is {method!r}, not gptq or awq")
    if int(quant.get("bits") or 0) != BITS:
        problems.append(f"bits is {quant.get('bits')!r}, expected {BITS}")
    if int(quant.get("group_size") or 0) != GROUP:
        problems.append(f"group_size is {quant.get('group_size')!r}, expected {GROUP}")
    files = safetensors_in(model_dir)
    if not files:
        return [*problems, "no safetensors files"]
    qweights = 0
    for path in files:
        header = read_header(path)
        names = {n for n in header if n != "__metadata__"}
        for name in names:
            if name.endswith(FORBIDDEN_SUFFIXES):
                problems.append(f"{name} is a layout the validators cannot read")
            if name.endswith(".qweight"):
                qweights += 1
                stem = name[: -len(".qweight")]
                if header[name].get("dtype") not in PACKED:
                    problems.append(f"{name} is {header[name].get('dtype')}, not packed int32")
                needed = ("qzeros", "scales", "g_idx") if method == "gptq" else ("qzeros", "scales")
                for part in needed:
                    if f"{stem}.{part}" not in names and not (part == "g_idx" and quant.get("desc_act") is False):
                        problems.append(f"{stem}.{part} is missing")
    if qweights == 0:
        problems.append("no .qweight tensors: the model is not quantised")
    return sorted(set(problems))


def calibration(pairs: Path, count: int, max_tokens: int) -> list[str]:
    from ccslm.common import tokenizer

    tok = tokenizer()
    texts: list[str] = []
    for pair in iter_jsonl(pairs):
        text = pair["prompt"] + pair["completion"]
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) > max_tokens:
            text = tok.decode(ids[-max_tokens:])
        texts.append(text)
        if len(texts) >= count:
            break
    return texts


def quantize(model_dir: Path, pairs: Path, out: Path, count: int, max_tokens: int) -> dict[str, Any]:
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as exc:
        raise SystemExit("gptqmodel is not installed; see scripts/setup_gpu.sh (optional quant venv)") from exc
    config = QuantizeConfig(bits=BITS, group_size=GROUP, desc_act=False, sym=True)
    model = GPTQModel.load(str(model_dir), config)
    model.quantize(calibration(pairs, count, max_tokens), batch_size=1)
    shutil.rmtree(out, ignore_errors=True)
    model.save(str(out))
    write_portable_config(out)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        if (model_dir / name).is_file():
            shutil.copyfile(model_dir / name, out / name)
    return {"source": str(model_dir), "method": "gptq", "bits": BITS, "group_size": GROUP, "calibration": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="GPTQ 4 bit and the validator format checks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    go = sub.add_parser("run")
    go.add_argument("--model", required=True)
    go.add_argument("--pairs", required=True)
    go.add_argument("--count", type=int, default=128)
    go.add_argument("--max-tokens", type=int, default=16_384)
    go.add_argument("--out", required=True)
    look = sub.add_parser("check")
    look.add_argument("--model", required=True)
    look.add_argument("--base", default="")
    args = parser.parse_args()
    if args.cmd == "run":
        info = quantize(Path(args.model), Path(args.pairs), Path(args.out), args.count, args.max_tokens)
        problems = check_format(Path(args.out))
        info["format_problems"] = problems
        write_json(Path(args.out) / "quant_report.json", info)
        log(json.dumps(info))
        return
    problems = check_format(Path(args.model))
    result: dict[str, Any] = {"format_problems": problems}
    if not problems:
        from microtensor_subnet.artifact.params import model_parameters
        from microtensor_subnet.validator.provenance import fingerprint_similarity

        from ccslm.common import base_dir

        base = Path(args.base) if args.base else base_dir()
        result["parameters"] = model_parameters(Path(args.model))
        result["fingerprint_similarity"] = round(fingerprint_similarity(Path(args.model), base), 6)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
