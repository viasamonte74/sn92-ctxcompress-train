"""Layer pruning for Granite 4.0 H Micro: drop the least useful Mamba layers, then heal with train.py.

Granite 4.0 H Micro has 40 layers: Mamba-2 everywhere except attention at 5, 15, 25 and 35. The
provenance fingerprint compares the singular values of the q/k/v/o projections in layers 0 to 7 by
tensor name, which here means the attention block of layer 5, so layers 0 to 7 are never touched and
keep their indices. Only Mamba layers with index >= 8 and below the last layer are candidates;
attention layers are kept because they carry the long range lookups this task depends on.

Importance is the block influence of ShortGPT: the mean angular distance between a layer's input and
output hidden states over calibration sequences at full length (the task is long context, so short
prompts would rank layers wrongly). A layer that barely turns the residual stream is cheap to drop.

What it buys, by the arena's own formulas: parameters (e_N, weight 0.2), the computed memory, which
counts the KV cache of every layer in num_hidden_layers (e_mem, 0.3), and prefill and decode time
(e_TTFT 0.4, e_TPOT 0.1). It costs quality; treat every pruned variant as an experiment that must beat
the unpruned one on the estimated SLM score.

  python -m ccslm.prune score --model <dir> --pairs out/data/val/pairs.jsonl --samples 32 --out out/prune
  python -m ccslm.prune cut --model <dir> --scores out/prune/scores.json --drop 6 --out out/prune/drop6
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from ccslm.common import MAMBA_TYPES, base_dir, iter_jsonl, log, portable_config, write_json

PROTECTED_BELOW = 8
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


def candidates(config: dict[str, Any]) -> list[int]:
    types = config["layer_types"]
    last = config["num_hidden_layers"] - 1
    return [i for i, kind in enumerate(types) if kind in MAMBA_TYPES and PROTECTED_BELOW <= i < last]


def block_influence(model_dir: Path, pairs: Path, samples: int, max_tokens: int, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from ccslm.common import tokenizer

    tok = tokenizer(str(model_dir)) if (model_dir / "tokenizer.json").is_file() else tokenizer()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(device)
    model.eval()
    layers = model.model.layers
    totals = [0.0] * len(layers)
    counts = [0] * len(layers)

    def hook(index: int) -> Any:
        def record(module: Any, args: Any, kwargs: Any, output: Any) -> None:
            before = args[0] if args else kwargs["hidden_states"]
            after = output[0] if isinstance(output, tuple) else output
            cosine = torch.nn.functional.cosine_similarity(before.float(), after.float(), dim=-1)
            angle = torch.arccos(cosine.clamp(-1.0, 1.0)) / math.pi
            totals[index] += float(angle.sum())
            counts[index] += int(angle.numel())

        return record

    handles = [layer.register_forward_hook(hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    used = 0
    with torch.inference_mode():
        for pair in iter_jsonl(pairs):
            ids = tok.encode(pair["prompt"] + pair["completion"], add_special_tokens=False).ids[:max_tokens]
            model(input_ids=torch.tensor([ids], device=device), use_cache=False, logits_to_keep=1)
            used += 1
            if used >= samples:
                break
    for handle in handles:
        handle.remove()
    scores = [totals[i] / counts[i] if counts[i] else 0.0 for i in range(len(layers))]
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return {
        "model": str(model_dir),
        "samples": used,
        "max_tokens": max_tokens,
        "scores": scores,
        "layer_types": config["layer_types"],
        "candidates": candidates(config),
        "ranked": sorted(candidates(config), key=lambda i: scores[i]),
    }


def cut(model_dir: Path, drop: list[int], out: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    allowed = set(candidates(config))
    bad = [i for i in drop if i not in allowed]
    if bad:
        raise ValueError(f"layers {bad} may not be dropped; candidates are {sorted(allowed)}")
    total = config["num_hidden_layers"]
    kept = [i for i in range(total) if i not in set(drop)]
    renumber = {old: new for new, old in enumerate(kept)}
    tensors: dict[str, Any] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open has no __iter__
                found = LAYER.match(name)
                if found:
                    index = int(found.group(1))
                    if index not in renumber:
                        continue
                    name_out = f"model.layers.{renumber[index]}.{found.group(2)}"
                else:
                    name_out = name
                tensors[name_out] = handle.get_tensor(name).contiguous()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shards: list[dict[str, Any]] = [{}]
    size = 0
    for name in sorted(tensors):
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if size and size + nbytes > 4_000_000_000:
            shards.append({})
            size = 0
        shards[-1][name] = tensors[name]
        size += nbytes
    weight_map: dict[str, str] = {}
    for n, shard in enumerate(shards, 1):
        file = f"model-{n:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else "model.safetensors"
        save_file(shard, str(out / file), metadata={"format": "pt"})
        weight_map.update({name: file for name in shard})
    if len(shards) > 1:
        total_size = sum(t.numel() * t.element_size() for t in tensors.values())
        write_json(out / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    config["num_hidden_layers"] = len(kept)
    config["layer_types"] = [config["layer_types"][i] for i in kept]
    reference = json.loads((base_dir() / "config.json").read_text(encoding="utf-8"))
    write_json(out / "config.json", portable_config(config, reference))
    for name in COPY_FILES:
        if (model_dir / name).is_file():
            shutil.copyfile(model_dir / name, out / name)
    return {"source": str(model_dir), "dropped": sorted(drop), "kept": kept, "layers": len(kept)}


def fewest_to(model_dir: Path, ranked: list[int], max_params: int) -> list[int]:
    """The shortest prefix of the ranking whose removal brings the parameter count to max_params."""
    from ccslm.estimate import layer_parameters

    outside, per = layer_parameters(model_dir)
    total = outside + sum(per.values())
    drop: list[int] = []
    for index in ranked:
        if total <= max_params:
            break
        drop.append(index)
        total -= per[index]
    if total > max_params:
        raise ValueError(f"dropping every candidate leaves {total:,} parameters, above {max_params:,}")
    return drop


def report(out: Path, base_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    from microtensor_subnet.artifact.params import model_parameters
    from microtensor_subnet.validator.provenance import fingerprint_similarity

    info = dict(info)
    info["parameters"] = model_parameters(out)
    info["parameters_base"] = model_parameters(base_dir)
    info["fingerprint_similarity"] = round(fingerprint_similarity(out, base_dir), 6)
    write_json(out / "prune_report.json", info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="prune Mamba layers of Granite 4.0 H Micro")
    sub = parser.add_subparsers(dest="cmd", required=True)
    score = sub.add_parser("score", help="block influence per layer on calibration pairs")
    score.add_argument("--model", required=True)
    score.add_argument("--pairs", required=True)
    score.add_argument("--samples", type=int, default=32)
    score.add_argument("--max-tokens", type=int, default=100_000)
    score.add_argument("--device", default="")
    score.add_argument("--out", required=True)
    chop = sub.add_parser("cut", help="drop the lowest scoring candidate layers")
    chop.add_argument("--model", required=True)
    chop.add_argument("--scores", default="")
    chop.add_argument("--drop", type=int, default=0, help="how many of the lowest ranked layers")
    chop.add_argument("--layers", default="", help="explicit comma separated layer indices instead")
    chop.add_argument(
        "--max-params",
        type=int,
        default=0,
        help="drop the fewest lowest ranked layers that bring the model to this many parameters "
        "(3000000000 moves Granite H Micro into the 3B price class)",
    )
    chop.add_argument("--base", default="", help="base model folder for the fingerprint (default: pinned)")
    chop.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "score":
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        found = block_influence(Path(args.model), Path(args.pairs), args.samples, args.max_tokens, device)
        write_json(Path(args.out) / "scores.json", found)
        log("lowest influence first: " + ", ".join(f"{i}:{found['scores'][i]:.4f}" for i in found["ranked"][:12]))
        return
    if args.layers:
        drop = [int(v) for v in args.layers.split(",") if v.strip()]
    else:
        ranked = json.loads(Path(args.scores).read_text(encoding="utf-8"))["ranked"]
        drop = ranked[: args.drop]
        if args.max_params:
            drop = fewest_to(Path(args.model), ranked, args.max_params)
    info = cut(Path(args.model), drop, Path(args.out))
    info = report(Path(args.out), Path(args.base) if args.base else base_dir(), info)
    log(json.dumps(info))


if __name__ == "__main__":
    main()
