"""Supervised fine tuning of Granite 4.0 H Micro on {prompt, completion} pairs, loss on the completion only.

The prompt is the arena's own slm_prompt with the item's system and user fields, tokenised exactly
as vLLM tokenises a raw completion request (Granite adds no BOS), and the completion ends with
<|end_of_text|>, the arena's stop token. Sessions run to ~120k tokens, so the forward pass keeps
only the logits of the completion positions (`logits_to_keep`), which is what makes a 100k token
example fit: the hidden states are checkpointed per layer and the 100k x 100k vocabulary logits are
never built. Batch size is one example; gradients accumulate over `--accumulate` examples.

LoRA targets the attention projections, the Mamba input projection and the shared MLP. The Mamba
`out_proj` is left out: with the mamba_ssm kernels the training path fuses it and reads its raw
weight, so an adapter there would never train. `--method full` trains every weight.

  python -m ccslm.train --pairs out/data/train/pairs.jsonl --eval out/data/val/pairs.jsonl \
      --out out/runs/r1 --method lora --rank 64 --epochs 1
  python -m ccslm.train ... --resume            # continue from out/runs/r1/checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

from ccslm.common import BASE, BASE_REVISION, iter_jsonl, log, tokenizer, write_json, write_portable_config

LORA_TARGETS = r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mamba\.in_proj|shared_mlp\.(input_linear|output_linear))$"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)
EOS = "<|end_of_text|>"


def encode_pairs(path: Path, model_dir: Path, max_tokens: int) -> list[dict[str, Any]]:
    tok = tokenizer(str(model_dir)) if (model_dir / "tokenizer.json").is_file() else tokenizer()
    eos = tok.token_to_id(EOS)
    rows = []
    for pair in iter_jsonl(path):
        prompt = tok.encode(pair["prompt"], add_special_tokens=False).ids
        completion = tok.encode(pair["completion"], add_special_tokens=False).ids + [eos]
        if len(prompt) + len(completion) > max_tokens:
            continue
        rows.append({"id": pair["item_id"], "prompt": prompt, "completion": completion})
    return rows


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def load_model(model_dir: Path, method: str, rank: int, device: str, dtype_name: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    import importlib.util

    if device.startswith("cuda"):
        try:
            import causal_conv1d  # noqa: F401
            import mamba_ssm  # noqa: F401
        except ImportError as exc:
            # transformers would fall back to its reference PyTorch scan, which runs out of memory
            # on a single 16k token example; see the kernel block in scripts/setup_gpu.sh.
            raise SystemExit(f"Mamba training kernels missing ({exc}); run scripts/setup_gpu.sh") from exc

    flash = importlib.util.find_spec("flash_attn") is not None
    attention = "flash_attention_2" if flash and device.startswith("cuda") else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype, attn_implementation=attention)
    model.config.use_cache = False
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if method == "lora":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
    return model


def completion_loss(model: Any, example: dict[str, Any], device: str) -> tuple[Any, int]:
    import torch
    import torch.nn.functional as F

    ids = example["prompt"] + example["completion"]
    keep = len(example["completion"]) + 1
    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids=input_ids, logits_to_keep=keep, use_cache=False)
    logits = out.logits[0, :-1].float()
    target = torch.tensor(example["completion"], device=device)
    return F.cross_entropy(logits, target, reduction="sum"), len(example["completion"])


def evaluate(model: Any, rows: list[dict[str, Any]], device: str) -> float:
    import torch

    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for example in rows:
            loss, n = completion_loss(model, example, device)
            total += float(loss)
            tokens += n
    model.train()
    return total / max(1, tokens)


class MasterAdam:
    """AdamW on fp32 master copies of bf16 weights, for full fine tuning.

    AdamW applied to bf16 weights directly loses most updates: a step of lr * m / sqrt(v) is about
    1e-4 of a weight, while bf16 resolves about 4e-3, so the update rounds away. The masters take the
    fp32 update and the bf16 weights are refreshed from them after each step. With bitsandbytes the
    moments are 8 bit (~6 GB instead of ~26 GB for 3.2B weights), which keeps a full fine tune with
    100k token examples inside 80 GB.
    """

    def __init__(self, params: list[Any], lr: float, eight_bit: bool) -> None:
        import torch

        self.params = list(params)
        self.masters = [p.detach().float().clone() for p in self.params]
        self.eight_bit = False
        if eight_bit:
            try:
                import bitsandbytes as bnb

                self.inner = bnb.optim.AdamW8bit(self.masters, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
                self.eight_bit = True
            except ImportError:
                log("bitsandbytes is not installed: fp32 AdamW moments (~26 GB more for 3.2B weights)")
        if not self.eight_bit:
            self.inner = torch.optim.AdamW(self.masters, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        self.param_groups = self.inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        import torch

        for p, m in zip(self.params, self.masters, strict=True):
            m.grad = None if p.grad is None else p.grad.detach().float()
        self.inner.step()
        with torch.no_grad():
            for p, m in zip(self.params, self.masters, strict=True):
                p.copy_(m)
                m.grad = None

    def state_dict(self) -> dict[str, Any]:
        return {"inner": self.inner.state_dict(), "masters": [m.detach().cpu() for m in self.masters], "eight_bit": self.eight_bit}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        import torch

        self.inner.load_state_dict(state["inner"])
        with torch.no_grad():
            for p, m, saved in zip(self.params, self.masters, state["masters"], strict=True):
                m.copy_(saved.to(m.device))
                p.copy_(m)


def save_checkpoint(model: Any, optimizer: Any, state: dict[str, Any], where: Path, method: str) -> None:
    import torch

    temp = where.with_name(where.name + ".tmp")
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True)
    if method == "lora":
        model.save_pretrained(str(temp))
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp / "model.pt")
    torch.save(optimizer.state_dict(), temp / "optimizer.pt")
    write_json(temp / "state.json", state)
    shutil.rmtree(where, ignore_errors=True)
    temp.rename(where)


def save_snapshot(model: Any, state: dict[str, Any], run: Path, method: str, keep: int) -> Path:
    """A numbered weights-only checkpoint under run/checkpoints/step-NNNNN, kept for later export.

    LoRA saves the adapter (small); full saves the bf16 weights. `keep` > 0 deletes all but the
    newest `keep` snapshots; 0 keeps every one. `python -m ccslm.export` merges any of them.
    """
    root = run / "checkpoints"
    where = root / f"step-{state['step']:05d}"
    temp = where.with_name(where.name + ".tmp")
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True)
    model.save_pretrained(str(temp), safe_serialization=True)
    write_json(temp / "state.json", {**state, "method": method, "saved_at": time.time()})
    shutil.rmtree(where, ignore_errors=True)
    temp.rename(where)
    if keep > 0:
        found = sorted(p for p in root.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
        for old in found[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    return where


def load_checkpoint(model: Any, optimizer: Any, where: Path, method: str) -> dict[str, Any]:
    import torch

    if method == "lora":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(where / "adapter_model.safetensors")))
    else:
        model.load_state_dict(torch.load(where / "model.pt", map_location="cpu"))
    optimizer.load_state_dict(torch.load(where / "optimizer.pt", map_location="cpu"))
    return json.loads((where / "state.json").read_text(encoding="utf-8"))


def export(model: Any, method: str, model_dir: Path, base_dir: Path, out: Path) -> None:
    """Merged bf16 safetensors, the base's tokenizer files byte for byte and its config.json."""
    import torch

    merged = model.merge_and_unload() if method == "lora" else model
    merged = merged.to(torch.bfloat16)
    merged.config.use_cache = True
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    for name in TOKENIZER_FILES:
        source = base_dir / name
        if source.is_file():
            shutil.copyfile(source, out / name)
    if (model_dir / "generation_config.json").is_file():
        shutil.copyfile(model_dir / "generation_config.json", out / "generation_config.json")
    write_portable_config(out, base_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="fine tune Granite 4.0 H Micro on ctx_compress pairs")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--eval", default="")
    parser.add_argument("--out", required=True, help="run folder: checkpoint, metrics, merged model")
    parser.add_argument("--model", default="", help="model folder to start from (default: the pinned base)")
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw8bit", help="full method: moments in fp32 or 8 bit (bitsandbytes)")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0, help="default 1e-4 lora, 1e-5 full")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=126_000)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=50, help="optimizer steps")
    parser.add_argument("--eval-items", type=int, default=24)
    parser.add_argument("--save-every", type=int, default=25, help="optimizer steps between numbered checkpoints")
    parser.add_argument("--keep-checkpoints", type=int, default=0, help="numbered checkpoints to keep (0 = all)")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps; the LR schedule still follows the full run")
    parser.add_argument("--schedule-steps", type=int, default=0, help="steps the cosine schedule spans (default: the full run of --epochs)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from ccslm.common import base_dir as pinned

    base_dir = Path(args.model) if args.model else pinned()
    # Tokenizer files always come from the pinned base so they stay byte identical to it.
    original_base = pinned()
    lr = args.lr or (1e-4 if args.method == "lora" else 1e-5)

    train_rows = encode_pairs(Path(args.pairs), base_dir, args.max_tokens)
    eval_rows = encode_pairs(Path(args.eval), base_dir, args.max_tokens)[: args.eval_items] if args.eval else []
    rng = random.Random(args.seed)
    order: list[int] = []
    for _ in range(max(1, math.ceil(args.epochs))):
        epoch = list(range(len(train_rows)))
        rng.shuffle(epoch)
        order += epoch
    order = order[: max(1, round(len(train_rows) * args.epochs))]
    run_steps = max(1, len(order) // args.accumulate)
    # The schedule spans the whole run so a short --max-steps run and its resumes follow one curve.
    schedule_steps = args.schedule_steps or run_steps
    total_steps = min(run_steps, args.max_steps) if args.max_steps else run_steps
    warmup = max(1, int(args.warmup * schedule_steps))
    log(
        f"{len(train_rows)} train pairs, {len(eval_rows)} eval pairs, stop at step {total_steps} of {run_steps}, "
        f"schedule over {schedule_steps} steps, lr {lr}"
    )

    model = load_model(base_dir, args.method, args.rank, device, args.dtype)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable parameters {sum(p.numel() for p in trainable):,}")
    if args.method == "full" and args.dtype != "fp32":
        optimizer: Any = MasterAdam(trainable, lr, eight_bit=args.optim == "adamw8bit")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    state = {"step": 0, "cursor": 0, "seen_tokens": 0, "train_pairs": len(train_rows), "schedule_steps": schedule_steps}
    checkpoint = out / "checkpoint"
    if args.resume and checkpoint.is_dir():
        state = load_checkpoint(model, optimizer, checkpoint, args.method)
        if state.get("train_pairs", len(train_rows)) != len(train_rows):
            raise SystemExit(
                f"resume: the checkpoint saw {state['train_pairs']} train pairs, now {len(train_rows)}; "
                "keep --pairs, --max-tokens and --seed as in the first run"
            )
        if not args.schedule_steps and state.get("schedule_steps"):
            schedule_steps = int(state["schedule_steps"])
            warmup = max(1, int(args.warmup * schedule_steps))
        state.update(train_pairs=len(train_rows), schedule_steps=schedule_steps)
        log(f"resumed at step {state['step']}")
    write_json(
        out / "config.json",
        {
            "base_model": BASE,
            "base_revision": BASE_REVISION,
            "start_from": str(base_dir),
            "method": args.method,
            "rank": args.rank if args.method == "lora" else None,
            "lora_targets": LORA_TARGETS if args.method == "lora" else None,
            "learning_rate": lr,
            "epochs": args.epochs,
            "accumulate": args.accumulate,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "pairs": str(args.pairs),
            "train_pairs": len(train_rows),
            "eval": str(args.eval),
        },
    )
    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    model.train()
    started = time.time()
    window_loss, window_tokens, window_seq = 0.0, 0, 0
    while state["step"] < total_steps:
        for group in param_groups(optimizer):
            group["lr"] = lr_at(state["step"], schedule_steps, lr, warmup)
        span = order[state["cursor"] : state["cursor"] + args.accumulate]
        skipped = state.setdefault("skipped", [])
        while True:
            # An example that runs out of memory is skipped for good (kept in the state, so a resume
            # skips it too) and the batch is redone without it: a partial backward leaves gradients
            # that cannot be told apart from the rest of the batch.
            optimizer.zero_grad(set_to_none=True)
            batch = [i for i in span if i not in skipped]
            tokens = sum(len(train_rows[i]["completion"]) for i in batch)
            window_loss, window_tokens, window_seq = 0.0, 0, 0
            failed = None
            for i in batch:
                try:
                    loss, n = completion_loss(model, train_rows[i], device)
                    (loss / tokens).backward()
                except torch.OutOfMemoryError:
                    failed = i
                    break
                window_loss += float(loss.detach())
                window_tokens += n
                window_seq += len(train_rows[i]["prompt"]) + n
            if failed is None:
                break
            loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped.append(failed)
            seq = len(train_rows[failed]["prompt"]) + len(train_rows[failed]["completion"])
            log(f"out of memory on {train_rows[failed]['id']} ({seq} tokens): skipped, redoing step {state['step'] + 1}")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        state["step"] += 1
        state["cursor"] += len(span)
        state["seen_tokens"] += window_seq
        row: dict[str, Any] = {
            "step": state["step"],
            "loss": round(window_loss / max(1, window_tokens), 5),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": round(float(norm), 4),
            "sequence_tokens": window_seq,
            "elapsed_s": round(time.time() - started, 1),
        }
        if device.startswith("cuda"):
            row["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        window_loss, window_tokens, window_seq = 0.0, 0, 0
        if eval_rows and (state["step"] % args.eval_every == 0 or state["step"] == total_steps):
            row["eval_loss"] = round(evaluate(model, eval_rows, device), 5)
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        log(json.dumps(row))
        if state["step"] % args.save_every == 0 or state["step"] == total_steps:
            snapshot = save_snapshot(model, {**state, **{k: row[k] for k in ("loss", "eval_loss") if k in row}}, out, args.method, args.keep_checkpoints)
            save_checkpoint(model, optimizer, state, checkpoint, args.method)
            log(f"checkpoint {snapshot}")
    metrics.close()
    write_json(
        out / "env.json",
        {
            "libraries": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "peft": __import__("peft").__version__,
            },
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "python": platform.python_version(),
            "hours": round((time.time() - started) / 3600, 3),
        },
    )
    if not args.no_export:
        export(model, args.method, base_dir, original_base, out / "merged")
        log(f"merged model at {out / 'merged'}")


def param_groups(optimizer: Any) -> list[dict[str, Any]]:
    return list(optimizer.param_groups)


if __name__ == "__main__":
    main()
