"""Shared paths, the live arena definition, jsonl helpers, the Granite tokenizer and stage markers."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ARENA_ID = "ctx_compress"
ARENA_URL = f"https://api.microtensor.cloud/v2/arenas/{ARENA_ID}"
BASE = "ibm-granite/granite-4.0-h-micro"
BASE_REVISION = "d5f01a3ea75f088947be3aae039f4ad52837dfde"
PROBE_HF = "Qwen/Qwen2.5-1.5B-Instruct"
PROBE_HF_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
HEADINGS = (
    "Primary request and intent",
    "Key technical concepts",
    "Files and code sections",
    "Errors and fixes",
    "Problem solving",
    "All user messages",
    "Pending tasks",
    "Current work",
    "Next step",
)
# The values the project was built against; `check_definition` warns when the live arena moves.
EXPECTED = {
    "allowed_bases": [BASE],
    "slm_prompt": (
        "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    ),
    "stop": ["<|end_of_text|>"],
    "max_tokens": 3072,
    "pass_threshold": 0.6,
}


def out_dir() -> Path:
    path = Path(os.environ.get("CCSLM_OUT") or ROOT / "out")
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temp.replace(target)
    return count


def write_json(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path | str, default: Any = None) -> Any:
    target = Path(path)
    if not target.is_file():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def definition(refresh: bool = False) -> dict[str, Any]:
    """The arena definition, cached under out/arena.json; refresh re-reads the live API."""
    cache = out_dir() / "arena.json"
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))["definition"]
    with urllib.request.urlopen(ARENA_URL, timeout=30) as response:  # noqa: S310
        document = json.loads(response.read().decode("utf-8"))
    write_json(cache, document)
    return document["definition"]


def check_definition(found: dict[str, Any]) -> list[str]:
    """Differences between the live definition and what this project was built for."""
    grading = found.get("grading") or {}
    generation = found.get("generation") or {}
    live = {
        "allowed_bases": found.get("allowed_bases"),
        "slm_prompt": found.get("slm_prompt"),
        "stop": generation.get("stop"),
        "max_tokens": generation.get("max_tokens"),
        "pass_threshold": grading.get("pass_threshold"),
    }
    problems = [f"{key}: built for {EXPECTED[key]!r}, live {value!r}" for key, value in live.items() if value != EXPECTED[key]]
    headings = grading.get("headings")
    if headings and tuple(headings) != HEADINGS:
        problems.append(f"headings changed: {headings}")
    return problems


def render_prompt(template: str, task: dict[str, Any]) -> str:
    """The validator's render_prompt: replace {key} with each task field, in task order."""
    rendered = template
    for key, value in task.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered


def slm_prompt(task: dict[str, Any]) -> str:
    return render_prompt(EXPECTED["slm_prompt"], task)


@lru_cache(maxsize=4)
def tokenizer(name: str = BASE, revision: str = BASE_REVISION) -> Any:
    from tokenizers import Tokenizer

    local = Path(name)
    if (local / "tokenizer.json").is_file():
        return Tokenizer.from_file(str(local / "tokenizer.json"))
    return Tokenizer.from_pretrained(name, revision=revision)


def count_tokens(text: str) -> int:
    """Token count as the grader counts the output: Granite tokenizer, default special token handling."""
    return len(tokenizer().encode(text).ids)


def base_dir() -> Path:
    """The local snapshot of the pinned base model, downloaded on first use.

    CCSLM_BASE_DIR points somewhere else; the tests use it for a tiny model of the same architecture.
    """
    override = os.environ.get("CCSLM_BASE_DIR")
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(BASE, revision=BASE_REVISION))


MAMBA_TYPES = ("mamba", "linear_attention")
ATTENTION_TYPES = ("attention", "full_attention")


def portable_config(model_config: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """The base's own config.json with this model's layer list, in the base's naming.

    transformers 5 renames Granite's layer types (mamba -> linear_attention, attention ->
    full_attention), writes time_step_limit as {"__float__": "Infinity"} and swaps torch_dtype for
    dtype when it saves. The serving engine and the validators load the file with their own
    versions, so a shipped model keeps the base's config and changes only what pruning changed
    (num_hidden_layers, layer_types) and what quantisation adds (quantization_config).
    """
    out = dict(reference)
    types = model_config.get("layer_types") or reference["layer_types"]
    out["layer_types"] = [
        "mamba" if t in MAMBA_TYPES else "attention" if t in ATTENTION_TYPES else t for t in types
    ]
    out["num_hidden_layers"] = len(out["layer_types"])
    if model_config.get("quantization_config"):
        out["quantization_config"] = model_config["quantization_config"]
    return out


def write_portable_config(model_dir: Path, reference_dir: Path | None = None) -> None:
    reference = json.loads(((reference_dir or base_dir()) / "config.json").read_text(encoding="utf-8"))
    found = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    write_json(model_dir / "config.json", portable_config(found, reference))


class Stages:
    """Done markers under out/stages so the pipeline can resume."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or out_dir()) / "stages"
        self.root.mkdir(parents=True, exist_ok=True)

    def done(self, name: str) -> bool:
        return (self.root / f"{name}.done").is_file()

    def mark(self, name: str, info: dict[str, Any] | None = None) -> None:
        write_json(self.root / f"{name}.done", {"finished": time.time(), **(info or {})})

    def clear(self, name: str) -> None:
        (self.root / f"{name}.done").unlink(missing_ok=True)
