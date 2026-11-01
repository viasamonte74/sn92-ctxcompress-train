"""Targets pass the grader's deterministic checks and state the probe answers the label expects."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ccslm import data, targets
from ccslm.common import read_jsonl


@pytest.fixture(scope="module")
def items(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    out = tmp_path_factory.mktemp("items")
    data.build_split("train", "pytest-targets", 4, Path(out), workers=4)
    return read_jsonl(Path(out) / "items.jsonl")


def probe(item: dict, kind: str, question: str = "") -> dict | None:
    for p in item["label"]["probes"]:
        if p["type"] == kind and question in p["question"]:
            return p
    return None


def test_every_target_passes_the_deterministic_checks(items: list[dict]) -> None:
    assert len(items) >= 8
    for item in items:
        summary, tokens, _ = targets.write_target(item)
        check = targets.verify(item, summary)
        assert check["valid"], item["item_id"]
        assert check["within_budget"] and tokens <= targets.target_tokens(item["label"])
        assert check["recall"] == 1.0, item["item_id"]
        assert check["invented"] == 0.0, (item["item_id"], check["invented_names"])
        assert summary.startswith("## Primary request and intent\n")


def test_targets_state_the_probe_answers(items: list[dict]) -> None:
    from microtensor_subnet.validator import compress

    for item in items:
        summary, _, _ = targets.write_target(item)
        found = compress.sections(summary)
        assert found is not None
        current = found["Current work"].lower()
        edited = probe(item, "exact", "edit most recently")
        if edited:
            assert any(a in current for a in edited["answers"])
        tested = probe(item, "word", "most recent test run")
        if tested:
            assert f"latest test run: {tested['answers'][0]}" in current
        error = probe(item, "exact", "exception type")
        if error:
            assert any(a in current for a in error["answers"])
        nxt = probe(item, "next_step")
        if nxt:
            line = found["Next step"]
            assert re.match(r"^Next action: \w+", line)
            said = line.removeprefix("Next action: ")
            kind_ok = nxt["answers"]["kind"] in compress.KINDS
            file_shown = len(said.split()) > 1
            expected = (0.5 * kind_ok + 0.5 * file_shown) if nxt["answers"]["file"] else float(kind_ok)
            if kind_ok and file_shown:
                expected = 1.0
            elif not kind_ok and not file_shown:
                expected = 0.0
            assert compress.probe_score(nxt, said) == expected, (item["item_id"], said, nxt["answers"])


def test_trimming_keeps_every_fact() -> None:
    oracle = {
        "issue": "parse_rows drops the last row. It lives in src/pkg/core.py; run tests/test_core.py.",
        "user_messages": ["parse_rows drops the last row. It lives in src/pkg/core.py; run tests/test_core.py."],
        "events": [{"kind": "read", "name": "Read", "path": f"src/pkg/m{i}.py", "args": "", "note": "", "status": "done", "errors": []} for i in range(40)],
        "opened": [f"src/pkg/m{i}.py" for i in range(40)],
        "created": [],
        "edited": [],
        "errors": [],
        "frames": [],
        "symbols": [],
        "tests": [],
        "next": {"kind": "read", "path": "src/pkg/m40.py"},
        "goal": "",
    }
    label = {
        "budget_tokens": 1000,
        "facts": [{"kind": "file", "value": f"src/pkg/m{i}.py", "aliases": [f"src/pkg/m{i}.py"]} for i in range(12)],
    }
    session = "\n".join(f"src/pkg/m{i}.py" for i in range(41)) + "\nparse_rows tests/test_core.py src/pkg/core.py"
    item = {"oracle": oracle, "label": label, "task": {"user": session}}
    summary, tokens, level = targets.write_target(item)
    assert tokens <= targets.target_tokens(label)
    check = targets.verify(item, summary)
    assert check["recall"] == 1.0 and check["invented"] == 0.0 and check["valid"]
    assert level == 0 or "src/pkg/m39.py" not in summary.split("## Errors and fixes")[0]


def test_probe_prompt_is_the_validators() -> None:
    from microtensor_subnet.validator.sandbox import judge

    from ccslm import probe as probe_module

    assert probe_module.PROMPT == judge.PROMPT
    message = probe_module.messages("S", "Q")[1]["content"]
    assert message == "<summary>\nS\n</summary>\n\nQuestion: Q"
