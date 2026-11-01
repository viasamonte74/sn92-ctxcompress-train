"""Merge a numbered training checkpoint into a servable bf16 model, like the end of `ccslm.train`.

  python -m ccslm.export --run out/runs/A --list                  # checkpoints with step, loss, eval loss
  python -m ccslm.export --run out/runs/A --step 100              # -> out/runs/A/merged-step-00100
  python -m ccslm.export --run out/runs/A --step latest --out /somewhere/model

The output has the base's tokenizer files byte for byte and its own config.json, so the pipeline's
eval, speed and package stages take it as they take `merged` (`--step` on the pipeline).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ccslm.common import base_dir as pinned
from ccslm.common import log, read_json


def snapshots(run: Path) -> list[Path]:
    root = run / "checkpoints"
    return sorted(p for p in root.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))


def find(run: Path, step: str) -> Path:
    found = snapshots(run)
    if not found:
        raise SystemExit(f"no checkpoints under {run / 'checkpoints'}")
    if step == "latest":
        return found[-1]
    where = run / "checkpoints" / f"step-{int(step):05d}"
    if not where.is_dir():
        raise SystemExit(f"no checkpoint {where.name}; have {', '.join(p.name for p in found)}")
    return where


def merged_dir(run: Path, step: int) -> Path:
    return run / f"merged-step-{step:05d}"


def export_snapshot(run: Path, snapshot: Path, out: Path, device: str = "cpu") -> Path:
    import torch
    from transformers import AutoModelForCausalLM

    from ccslm.train import export

    config = read_json(run / "config.json", {})
    state = read_json(snapshot / "state.json", {})
    method = state.get("method") or config.get("method") or "lora"
    start = Path(config["start_from"]) if config.get("start_from") else pinned()
    if method == "lora":
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(str(start), dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, str(snapshot))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=torch.bfloat16)
    model.to(device)
    export(model, method, start, pinned(), out)
    log(f"step {state.get('step')} ({method}) merged into {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="merge a numbered checkpoint into a servable model")
    parser.add_argument("--run", required=True, help="the training run folder, e.g. out/runs/A")
    parser.add_argument("--step", default="latest", help="step number or latest")
    parser.add_argument("--out", default="", help="default: <run>/merged-step-NNNNN")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if args.list:
        for snap in snapshots(run):
            state: dict[str, Any] = read_json(snap / "state.json", {})
            merged = merged_dir(run, int(state.get("step", 0)))
            print(json.dumps({"checkpoint": snap.name, **{k: state.get(k) for k in ("step", "loss", "eval_loss", "seen_tokens")}, "merged": merged.is_dir()}))
        return
    snapshot = find(run, args.step)
    step = int(read_json(snapshot / "state.json", {}).get("step") or snapshot.name.split("-")[-1])
    export_snapshot(run, snapshot, Path(args.out) if args.out else merged_dir(run, step), args.device)


if __name__ == "__main__":
    main()
