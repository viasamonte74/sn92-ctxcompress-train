"""What a smaller model buys, by the arena's own formulas, before any GPU time is spent.

Three things move when the model shrinks:

  SLM pool    E = 0.4 e_TTFT + 0.3 e_mem + 0.2 e_N + 0.1 e_TPOT against the base, score x (1 + 0.25 E)
  cost        the system and escalation pools price every SLM token by parameter class
              (definition r_class: <= 1B 0.02, <= 3B 0.05, else 0.1 micro-dollars per token),
              against frontier answers at their list prices (Claude Haiku 5.5 0.1/0.5, GLM 5.3 Flash
              0.048/0.2 dollars per million tokens in/out)
  speed       prefill FLOPs (first token time) and the bytes read per decoded token (time per token)

Parameters and computed memory are exact (safetensors headers, the validators' formula). The speed
ratios are first order estimates from FLOPs and bytes; the speed stage measures the real ones.

  python -m ccslm.estimate plan --scores out/prune/scores.json      # drop 0..10 Mamba layers, bf16 and GPTQ
  python -m ccslm.estimate model --model out/runs/A/merged
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from ccslm.common import MAMBA_TYPES, base_dir, definition, log, read_json

MAX_LEN = 122_880 + 3_072
BYTES = {"bf16": 2.0, "fp16": 2.0, "gptq": 0.5, "awq": 0.5}
R_CLASS = {"1b": 0.02, "3b": 0.05, "8b": 0.1}
FRONTIER = {"claude-haiku-5-5": (0.1, 0.5), "glm-5-3-flash": (0.048, 0.2)}
WEIGHTS = {"ttft": 0.4, "memory": 0.3, "parameters": 0.2, "tpot": 0.1}
LAYER = re.compile(r"^model\.layers\.(\d+)\.")


def layer_parameters(model_dir: Path) -> tuple[int, dict[int, int]]:
    """Parameters outside the layers and per layer index, from the safetensors headers."""
    from microtensor_subnet.artifact.params import _counted, read_header, safetensors_in

    outside = 0
    per: dict[int, int] = {}
    for path in safetensors_in(model_dir):
        for name, entry in read_header(path).items():
            if not isinstance(entry, dict):
                continue
            count = _counted(name, entry)
            found = LAYER.match(name)
            if found:
                per[int(found.group(1))] = per.get(int(found.group(1)), 0) + count
            else:
                outside += count
    return outside, per


def prices() -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    try:
        found = definition()
    except Exception:  # noqa: BLE001 - offline: the published values
        return dict(R_CLASS), dict(FRONTIER)
    r_class = {k: float(v) for k, v in (found.get("r_class") or R_CLASS).items()}
    frontier = {
        str(b["model"]): (float(b["usd_in_per_mtok"]), float(b["usd_out_per_mtok"]))
        for b in found.get("baseline_models") or []
        if "usd_in_per_mtok" in b
    } or dict(FRONTIER)
    return r_class, frontier


def token_price(parameters: int, r_class: dict[str, float]) -> float:
    """validator/context.py token_price: the class is chosen by the parameter count."""
    if parameters <= 1_000_000_000:
        return r_class.get("1b", 0.0)
    if parameters <= 3_000_000_000:
        return r_class.get("3b", 0.0)
    return r_class.get("8b", 0.0)


def profile(config: dict[str, Any], outside: int, per: dict[int, int], *, drop: tuple[int, ...] = (), fmt: str = "bf16") -> dict[str, Any]:
    kept = [i for i in sorted(per) if i not in set(drop)]
    types = [config["layer_types"][i] for i in kept]
    parameters = outside + sum(per[i] for i in kept)
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    kv_formula = 2 * len(kept) * kv_heads * head_dim * MAX_LEN * 2
    attention_layers = sum(1 for t in types if t not in MAMBA_TYPES)
    embedding = int(config["vocab_size"]) * hidden
    return {
        "layers": len(kept),
        "dropped": sorted(drop),
        "format": fmt,
        "parameters": parameters,
        "memory_bytes": int(parameters * BYTES[fmt]) + kv_formula,
        "matmul_parameters": parameters - embedding,
        "attention_layers": attention_layers,
        "hidden": hidden,
        "kv_per_token_bytes": attention_layers * 2 * kv_heads * head_dim * 2,
        "weight_bytes": parameters * BYTES[fmt],
    }


def prefill_flops(p: dict[str, Any], tokens: int) -> float:
    # weights once per token, plus causal attention (QK and AV) in the attention layers
    return 2.0 * p["matmul_parameters"] * tokens + 2.0 * p["attention_layers"] * tokens * tokens * p["hidden"]


def decode_bytes(p: dict[str, Any], context: int) -> float:
    return p["weight_bytes"] + p["kv_per_token_bytes"] * context


def gains(mine: dict[str, Any], ref: dict[str, Any], tokens_in: int, tokens_out: int) -> dict[str, float]:
    def gain(r: float, m: float) -> float:
        return min(1.0, max(0.0, (r - m) / r)) if r > 0 else 0.0

    found = {
        "ttft": gain(prefill_flops(ref, tokens_in), prefill_flops(mine, tokens_in)),
        "memory": gain(ref["memory_bytes"], mine["memory_bytes"]),
        "parameters": gain(ref["parameters"], mine["parameters"]),
        # a 4 bit kernel still dequantises, so decode gains less than the byte ratio; count half of it
        "tpot": gain(decode_bytes(ref, tokens_in), decode_bytes(mine, tokens_in)) * (0.5 if mine["format"] in ("gptq", "awq") else 1.0),
    }
    if mine["format"] in ("gptq", "awq"):
        found["ttft"] = min(found["ttft"], gain(prefill_flops(ref, tokens_in), prefill_flops({**mine, "format": "bf16"}, tokens_in)))
    found["E"] = sum(WEIGHTS[k] * found[k] for k in WEIGHTS)
    return found


def costs(parameters: int, tokens_in: int, tokens_out: int) -> dict[str, Any]:
    r_class, frontier = prices()
    price = token_price(parameters, r_class)
    mine = (tokens_in + tokens_out) * price
    row: dict[str, Any] = {"price_per_token_micros": price, "item_cost_micros": round(mine)}
    for name, (usd_in, usd_out) in frontier.items():
        theirs = tokens_in * usd_in + tokens_out * usd_out
        row[f"vs_{name}"] = round(theirs / mine, 2) if mine else None
        # the cost half of the system pool's v_f for this ratio (latency term left out)
        row[f"v_cost_{name}"] = round(min(1.0, max(0.0, 0.7 * math.log(theirs / mine) / math.log(10))), 3) if mine else None
    return row


def plan(model_dir: Path, scores: Path | None, max_drop: int, tokens_in: int, tokens_out: int) -> list[dict[str, Any]]:
    from ccslm.prune import candidates

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    outside, per = layer_parameters(model_dir)
    ranked = read_json(scores, {}).get("ranked") if scores else None
    order = list(ranked or sorted(candidates(config), reverse=True))
    ref = profile(config, outside, per)
    rows = []
    for k in range(0, min(max_drop, len(order)) + 1):
        for fmt in ("bf16", "gptq"):
            mine = profile(config, outside, per, drop=tuple(order[:k]), fmt=fmt)
            found = gains(mine, ref, tokens_in, tokens_out)
            rows.append(
                {
                    "drop": k,
                    "format": fmt,
                    "parameters": mine["parameters"],
                    "memory_gb": round(mine["memory_bytes"] / 1e9, 2),
                    "E": round(found["E"], 3),
                    "slm_multiplier": round(1 + 0.25 * found["E"], 3),
                    **{f"e_{k2}": round(v, 3) for k2, v in found.items() if k2 != "E"},
                    **costs(mine["parameters"], tokens_in, tokens_out),
                    "layers_dropped": mine["dropped"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="parameters, memory, E and cost by the arena's formulas")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="drop 0..N Mamba layers, bf16 and GPTQ")
    p_plan.add_argument("--model", default="", help="default: the pinned base")
    p_plan.add_argument("--scores", default="", help="prune.py scores.json for the drop order")
    p_plan.add_argument("--max-drop", type=int, default=10)
    p_model = sub.add_parser("model", help="one model folder against the base")
    p_model.add_argument("--model", required=True)
    for p in (p_plan, p_model):
        p.add_argument("--input-tokens", type=int, default=60_000, help="a typical hidden item is 35k-95k")
        p.add_argument("--output-tokens", type=int, default=720, help="compact targets average ~720")
        p.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cmd == "plan":
        rows = plan(Path(args.model) if args.model else base_dir(), Path(args.scores) if args.scores else None, args.max_drop, args.input_tokens, args.output_tokens)
        if args.json:
            print(json.dumps(rows, indent=2))
            return
        print(f"{'drop':>4} {'fmt':5} {'params':>13} {'mem GB':>7} {'E':>6} {'xSLM':>6} {'price':>6} {'vs Haiku':>8} {'vs GLM':>7}")
        for r in rows:
            print(
                f"{r['drop']:>4} {r['format']:5} {r['parameters']:>13,} {r['memory_gb']:>7} {r['E']:>6} {r['slm_multiplier']:>6} "
                f"{r['price_per_token_micros']:>6} {r['vs_claude-haiku-5-5']:>7}x {r['vs_glm-5-3-flash']:>6}x"
            )
        return
    model = Path(args.model)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    quant = (config.get("quantization_config") or {}).get("quant_method", "")
    base = base_dir()
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    mine = profile(config, *layer_parameters(model), fmt=quant if quant in BYTES else "bf16")
    ref = profile(base_config, *layer_parameters(base))
    found = {"model": str(model), **mine, **{k: round(v, 4) for k, v in gains(mine, ref, args.input_tokens, args.output_tokens).items()}}
    found.update(costs(mine["parameters"], args.input_tokens, args.output_tokens))
    print(json.dumps(found, indent=2))
    log(f"E ~ {found['E']}, SLM score x{1 + 0.25 * found['E']:.3f}, {found['price_per_token_micros']} micro-dollars per token")


if __name__ == "__main__":
    main()
