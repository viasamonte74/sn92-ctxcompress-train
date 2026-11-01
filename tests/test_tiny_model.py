"""train -> merge -> prune -> package -> validator checks on a tiny random Granite 4.0 H model, on CPU.

The tiny model keeps the real architecture (GraniteMoeHybrid, Mamba-2 layers with attention at 5),
the real tokenizer files and vocabulary, so every file the validators look at has the real shape.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parent.parent
LAYERS = 12
# With mamba_ssm and causal_conv1d installed transformers runs the Mamba layers on their CUDA
# kernels whatever the tensor's device, so the tiny model trains on the GPU when there is one.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTENTION = (5,)


def real_base() -> Path:
    from huggingface_hub import snapshot_download

    from ccslm.common import BASE, BASE_REVISION

    return Path(
        snapshot_download(
            BASE,
            revision=BASE_REVISION,
            allow_patterns=["*.json", "*.jinja", "*.txt"],
        )
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import AutoConfig, AutoModelForCausalLM

    source = real_base()
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    raw.update(
        hidden_size=64,
        intermediate_size=128,
        shared_intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=LAYERS,
        layer_types=["attention" if i in ATTENTION else "mamba" for i in range(LAYERS)],
        mamba_n_heads=8,  # in_proj width 296: causal_conv1d needs strides in multiples of 8
        mamba_d_head=16,
        mamba_d_state=16,
        mamba_chunk_size=32,
    )
    where = tmp_path_factory.mktemp("tiny_base")
    (where / "config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(where)), dtype=torch.float32)
    model.to(torch.bfloat16).save_pretrained(str(where), safe_serialization=True)
    # keep the on-disk config in the base's own naming, as the real snapshot has it
    (where / "config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, where / name)
    return where


def pairs_file(where: Path) -> Path:
    from ccslm.common import slm_prompt

    rows = []
    for i in range(6):
        task = {"system": "Summarise the session.", "user": f"<session>\n[user]\nfix bug {i} in src/pkg/m{i}.py\n</session>"}
        rows.append({"item_id": f"t{i}", "prompt": slm_prompt(task), "completion": f"## Primary request and intent\nFix src/pkg/m{i}.py"})
    path = where / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def run(args: list[str], env: dict[str, str]) -> None:
    done = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]


def test_train_merge_prune_package(tiny: Path, tmp_path: Path) -> None:
    from microtensor_subnet.validator.provenance import fingerprint_similarity
    from safetensors import safe_open

    env = {**os.environ, "CCSLM_BASE_DIR": str(tiny), "CCSLM_OUT": str(tmp_path / "out")}
    pairs = pairs_file(tmp_path)
    run_dir = tmp_path / "run"
    run(
        ["ccslm.train", "--pairs", str(pairs), "--eval", str(pairs), "--model", str(tiny), "--out", str(run_dir),
         "--method", "lora", "--rank", "4", "--accumulate", "2", "--max-steps", "2", "--save-every", "1",
         "--eval-every", "2", "--eval-items", "2", "--device", DEVICE, "--dtype", "fp32", "--lr", "5e-3"],
        env,
    )
    merged = run_dir / "merged"
    assert json.loads((merged / "config.json").read_text()) == json.loads((tiny / "config.json").read_text())
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2 and "eval_loss" in metrics[-1]
    assert (merged / "tokenizer.json").read_bytes() == (tiny / "tokenizer.json").read_bytes()
    similarity = fingerprint_similarity(merged, tiny)
    assert similarity >= 0.9

    # the adapter changed the attention and the Mamba input projection, never the Mamba out_proj
    def tensor(folder: Path, name: str) -> object:
        for path in folder.glob("*.safetensors"):
            with safe_open(str(path), framework="pt") as handle:
                if name in handle.keys():  # noqa: SIM118
                    return handle.get_tensor(name).float()
        raise KeyError(name)

    assert not torch.equal(tensor(merged, "model.layers.5.self_attn.q_proj.weight"), tensor(tiny, "model.layers.5.self_attn.q_proj.weight"))
    assert not torch.equal(tensor(merged, "model.layers.0.mamba.in_proj.weight"), tensor(tiny, "model.layers.0.mamba.in_proj.weight"))
    assert torch.equal(tensor(merged, "model.layers.0.mamba.out_proj.weight"), tensor(tiny, "model.layers.0.mamba.out_proj.weight"))

    # resume continues from the checkpoint instead of starting over
    run(
        ["ccslm.train", "--pairs", str(pairs), "--model", str(tiny), "--out", str(run_dir), "--method", "lora",
         "--rank", "4", "--accumulate", "2", "--max-steps", "3", "--device", DEVICE, "--dtype", "fp32", "--resume", "--no-export"],
        env,
    )
    steps = [json.loads(line)["step"] for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3]

    # every save left a numbered checkpoint, and any of them merges into a servable model
    assert sorted(p.name for p in (run_dir / "checkpoints").glob("step-*")) == ["step-00001", "step-00002", "step-00003"]
    run(["ccslm.export", "--run", str(run_dir), "--step", "2"], env)
    exported = run_dir / "merged-step-00002"
    assert json.loads((exported / "config.json").read_text()) == json.loads((tiny / "config.json").read_text())
    name = "model.layers.5.self_attn.q_proj.weight"
    assert torch.allclose(tensor(exported, name), tensor(merged, name), atol=2e-2)
    assert not torch.equal(tensor(exported, name), tensor(tiny, name))

    # prune two Mamba layers above 7 and check the result loads and fingerprints like the base
    pruned = tmp_path / "pruned"
    run(["ccslm.prune", "cut", "--model", str(merged), "--layers", "8,10", "--base", str(tiny), "--out", str(pruned)], env)
    report = json.loads((pruned / "prune_report.json").read_text())
    config = json.loads((pruned / "config.json").read_text())
    assert config["num_hidden_layers"] == LAYERS - 2
    assert config["layer_types"][5] == "attention"
    assert report["fingerprint_similarity"] >= 0.999
    for i in range(8):
        assert torch.equal(tensor(pruned, f"model.layers.{i}.input_layernorm.weight"), tensor(merged, f"model.layers.{i}.input_layernorm.weight"))
    assert torch.equal(tensor(pruned, "model.layers.8.mamba.in_proj.weight"), tensor(merged, "model.layers.9.mamba.in_proj.weight"))
    assert torch.equal(tensor(pruned, "model.layers.9.mamba.in_proj.weight"), tensor(merged, "model.layers.11.mamba.in_proj.weight"))
    model = transformers.AutoModelForCausalLM.from_pretrained(str(pruned), dtype=torch.float32).to(DEVICE)
    out = model(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=DEVICE))
    assert out.logits.shape[-1] == config["vocab_size"]

    # attention layers and the protected first eight may never be cut
    bad = subprocess.run(
        [sys.executable, "-m", "ccslm.prune", "cut", "--model", str(merged), "--layers", "5", "--base", str(tiny), "--out", str(tmp_path / "bad")],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert bad.returncode != 0 and "may not be dropped" in bad.stderr

    # package and run the validators' checks (bounds lowered for the tiny model only)
    from ccslm import package

    os.environ["CCSLM_BASE_DIR"] = str(tiny)
    try:
        repo = tmp_path / "slm_repo"
        package.build(merged, repo, run_dir=run_dir, eval_report=None, extra={})
        result = package.checks(repo, bounds=(1, 8_000_000_000))
    finally:
        os.environ.pop("CCSLM_BASE_DIR", None)
    assert result["ok"], result["problems"]
    assert result["preflight"]["message"] == "ok"
    recipe = json.loads((repo / "training" / "recipe.json").read_text())
    assert recipe["base_model"] == "ibm-granite/granite-4.0-h-micro"
    assert (repo / "training" / "runs" / "run" / "metrics.jsonl").read_text().strip()


def test_block_influence_scores_only_rank_candidates(tiny: Path, tmp_path: Path) -> None:
    env = {**os.environ, "CCSLM_BASE_DIR": str(tiny)}
    pairs = pairs_file(tmp_path)
    run(["ccslm.prune", "score", "--model", str(tiny), "--pairs", str(pairs), "--samples", "3", "--device", DEVICE, "--out", str(tmp_path / "scores")], env)
    found = json.loads((tmp_path / "scores" / "scores.json").read_text())
    assert len(found["scores"]) == LAYERS and found["samples"] == 3
    assert all(0.0 <= s <= 1.0 for s in found["scores"]) and any(s > 0 for s in found["scores"])
    assert sorted(found["ranked"]) == [8, 9, 10]


def test_full_fine_tune_updates_bf16_weights_through_fp32_masters(tiny: Path, tmp_path: Path) -> None:
    from safetensors import safe_open

    env = {**os.environ, "CCSLM_BASE_DIR": str(tiny)}
    pairs = pairs_file(tmp_path)
    run_dir = tmp_path / "full"
    common = ["ccslm.train", "--pairs", str(pairs), "--model", str(tiny), "--out", str(run_dir), "--method", "full",
              "--optim", "adamw", "--accumulate", "2", "--device", DEVICE, "--dtype", "bf16", "--lr", "1e-4"]
    run([*common, "--max-steps", "2", "--save-every", "1"], env)
    run([*common, "--max-steps", "3", "--resume"], env)
    steps = [json.loads(line)["step"] for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3]

    def load(folder: Path, name: str) -> object:
        for path in folder.glob("*.safetensors"):
            with safe_open(str(path), framework="pt") as handle:
                if name in handle.keys():  # noqa: SIM118
                    return handle.get_tensor(name).float()
        raise KeyError(name)

    # every weight class moved, including the Mamba out_proj that LoRA cannot reach
    for name in ("model.layers.0.mamba.out_proj.weight", "model.layers.0.shared_mlp.input_linear.weight"):
        assert not torch.equal(load(run_dir / "merged", name), load(tiny, name)), name
    assert json.loads((run_dir / "merged" / "config.json").read_text()) == json.loads((tiny / "config.json").read_text())


def test_master_adam_keeps_updates_that_bf16_rounds_away() -> None:
    from ccslm.train import MasterAdam

    plain = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    mastered = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    direct = torch.optim.AdamW([plain], lr=1e-3, weight_decay=0.0)
    wrapped = MasterAdam([mastered], lr=1e-3, eight_bit=False)
    for _ in range(10):
        for p, opt in ((plain, direct), (mastered, wrapped)):
            opt.zero_grad()
            p.grad = torch.ones_like(p)
            opt.step()
    assert torch.equal(plain.detach(), torch.ones(4, dtype=torch.bfloat16))  # 1e-3 steps vanish at 1.0 in bf16
    assert float(mastered.detach()[0]) < 0.995  # the fp32 masters accumulated ten of them


def test_max_params_drops_the_fewest_ranked_layers_and_estimate_plans(tiny: Path, tmp_path: Path) -> None:
    from ccslm.estimate import layer_parameters, plan, token_price
    from ccslm.prune import fewest_to

    outside, per = layer_parameters(tiny)
    total = outside + sum(per.values())
    ranked = [10, 8, 9]
    assert fewest_to(tiny, ranked, total) == []
    assert fewest_to(tiny, ranked, total - per[10]) == [10]
    assert fewest_to(tiny, ranked, total - per[10] - 1) == [10, 8]
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"ranked": ranked}))
    rows = plan(tiny, scores, 3, 1000, 100)
    bf16 = [r for r in rows if r["format"] == "bf16"]
    assert [r["drop"] for r in bf16] == [0, 1, 2, 3]
    assert all(a["parameters"] > b["parameters"] for a, b in zip(bf16, bf16[1:]))
    assert bf16[1]["layers_dropped"] == [10] and bf16[0]["E"] == 0.0 and bf16[-1]["E"] > 0
    assert token_price(3_191_396_096, {"1b": 0.02, "3b": 0.05, "8b": 0.1}) == 0.1
    assert token_price(2_962_847_168, {"1b": 0.02, "3b": 0.05, "8b": 0.1}) == 0.05
