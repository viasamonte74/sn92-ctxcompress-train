"""Training targets: the nine heading summary written from an item's oracle block.

Every identifier in a target is copied from text that is visible in the rendered session, so the
grader's hallucination rate is zero; every key fact of the label is listed, so recall is one; the
probe answers (last edit, latest test result, latest exception, next action, the task) are stated as
plain `name: value` lines the probe agent can read. The writer then trims low value detail until the
summary fits well inside the item's token budget. `verify` checks a summary with the grader's own
functions; `python -m ccslm.targets` writes {prompt, completion} pairs for training.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from ccslm.common import HEADINGS, count_tokens, iter_jsonl, log, slm_prompt, write_json, write_jsonl

SHARE = 0.85
CAP = 1400
MAX_USER_CHARS = 420
MIN_SOLVING_LINES = 3
SNAKE = re.compile(r"(?<![\w./-])([A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+)(?![\w.(/-])")
TEST_PATH = re.compile(r"(?<![\w/.-])(tests?/[\w./-]+\.py)")


def _first_sentence(text: str) -> str:
    return _clip(text.split(". ")[0].rstrip("."), 200)


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _command(event: dict[str, Any]) -> str:
    return event.get("args") or event.get("path") or event.get("name", "")


def _file_order(oracle: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for path in [*oracle["opened"], *oracle["created"], *oracle["edited"]]:
        if path not in files:
            files.append(path)
    changed = [p for p in files if p in oracle["edited"] or p in oracle["created"]]
    return changed + [p for p in files if p not in changed]


def _fact_files(label: dict[str, Any]) -> set[str]:
    return {f["value"] for f in label.get("facts") or [] if f.get("kind") == "file"}


def _file_lines(oracle: dict[str, Any], keep: set[str], extra: int) -> list[str]:
    tested: dict[str, bool] = {}
    for event in oracle["events"]:
        if event["kind"] == "test" and event.get("path"):
            tested[event["path"]] = bool(event.get("passed"))
    last_edit = oracle["edited"][-1] if oracle["edited"] else ""
    lines: list[str] = []
    extras = 0
    for path in _file_order(oracle):
        if path not in keep:
            if extras >= extra:
                continue
            extras += 1
        tags: list[str] = []
        if path in oracle["created"]:
            tags.append("created")
        if path in oracle["edited"]:
            tags.append("edited, most recent edit" if path == last_edit else "edited")
        if path in oracle["opened"]:
            tags.append("read")
        if path in tested:
            tags.append(f"tests run, latest {'pass' if tested[path] else 'fail'}")
        lines.append(f"- {path}: {', '.join(tags) or 'touched'}")
    return lines


def _error_lines(oracle: dict[str, Any], with_messages: bool) -> list[str]:
    events = oracle["events"]
    lines: list[str] = []
    for name, message in oracle["errors"]:
        where = -1
        for i, event in enumerate(events):
            if any(e[0] == name for e in event.get("errors") or []):
                where = i
        source = events[where] if where >= 0 else None
        text = f"- {name}"
        if with_messages and message:
            text += f": {_clip(message, 110)}"
        if source is not None:
            text += f" (from {_clip(_command(source), 80)})"
            later = [e for e in events[where + 1 :] if e["kind"] == "test"]
            same = [e for e in later if _command(e) == _command(source)]
            edits = [e for e in events[where + 1 :] if e["kind"] == "edit" and e.get("path")]
            if same and same[-1].get("passed"):
                text += "; fixed, a later run of the same command passed"
            elif edits:
                text += f"; after it the agent edited {edits[-1]['path']}"
                if same:
                    text += ", the same command still fails"
            else:
                text += "; not fixed yet"
        lines.append(text)
    return lines or ["- No errors so far."]


def _solving_lines(oracle: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    reads: list[str] = []

    def flush() -> None:
        if not reads:
            return
        shown = ", ".join(reads[:3]) + (f" and {len(reads) - 3} more" if len(reads) > 3 else "")
        lines.append(f"- Read {len(reads)} file{'s' if len(reads) > 1 else ''}: {shown}.")
        reads.clear()

    for event in oracle["events"]:
        kind = event["kind"]
        if kind == "read":
            if event.get("path") and event["path"] not in reads:
                reads.append(event["path"])
            continue
        flush()
        note = f" {event['note']}" if event.get("note") else ""
        if kind == "test":
            outcome = "pass" if event.get("passed") else "fail"
            result = f" ({event['result']})" if event.get("result") else ""
            lines.append(f"- Ran {_clip(_command(event), 90)}: {outcome}{result}.{note}")
        elif kind == "edit":
            lines.append(f"- Edited {event.get('path') or 'a file'}.{note}")
        elif kind == "search":
            lines.append(f"- Searched: {_clip(_command(event), 90)}.")
        else:
            lines.append(f"- {kind.capitalize()}: {_clip(_command(event), 90)}.{note}")
    flush()
    return lines


def _compact_file_lines(oracle: dict[str, Any], keep: set[str], extra: int) -> list[str]:
    """Files grouped by role on a few lines instead of one line per file."""
    order = _file_order(oracle)
    extras = [p for p in order if p not in keep]
    shown = set(keep) | set(extras[:extra])
    last_edit = oracle["edited"][-1] if oracle["edited"] else ""
    tested: dict[str, bool] = {}
    for event in oracle["events"]:
        if event["kind"] == "test" and event.get("path"):
            tested[event["path"]] = bool(event.get("passed"))
    lines: list[str] = []
    edited = [p for p in oracle["edited"] if p in shown]
    if edited:
        lines.append("- Edited: " + ", ".join(f"{p} (most recent edit)" if p == last_edit else p for p in dict.fromkeys(edited)))
    created = [p for p in oracle["created"] if p in shown]
    if created:
        lines.append("- Created: " + ", ".join(dict.fromkeys(created)))
    if tested:
        lines.append("- Tests run: " + ", ".join(f"{p} ({'pass' if ok else 'fail'})" for p, ok in tested.items()))
    read = [p for p in order if p in oracle["opened"] and p in shown]
    if read:
        lines.append("- Read: " + ", ".join(read))
    return lines or ["- None yet."]


def _compact_solving_lines(oracle: dict[str, Any]) -> list[str]:
    """Tests, edits and searches in order; reads are counted, their files are listed above."""
    lines: list[str] = []
    reads = sum(1 for e in oracle["events"] if e["kind"] == "read")
    searches = [_clip(_command(e), 60) for e in oracle["events"] if e["kind"] == "search"]
    if reads:
        lines.append(f"- Read {reads} times while investigating (files above).")
    if searches:
        lines.append("- Searched: " + "; ".join(dict.fromkeys(searches)))
    for event in oracle["events"]:
        kind = event["kind"]
        if kind == "test":
            outcome = "pass" if event.get("passed") else "fail"
            result = f" ({event['result']})" if event.get("result") else ""
            lines.append(f"- Ran {_clip(_command(event), 80)}: {outcome}{result}")
        elif kind == "edit":
            note = f" {event['note']}" if event.get("note") else ""
            lines.append(f"- Edited {event.get('path') or 'a file'}.{note}")
        elif kind not in ("read", "search"):
            lines.append(f"- {kind.capitalize()}: {_clip(_command(event), 80)}")
    return lines


def _concept_lines(oracle: dict[str, Any], with_commands: bool) -> list[str]:
    names: list[str] = []
    for found in SNAKE.findall(oracle["issue"]):
        if found not in names and not found.startswith("test_"):
            names.append(found)
    for symbol in oracle["symbols"]:
        if symbol not in names:
            names.append(symbol)
    for _, function in oracle["frames"]:
        if function not in names:
            names.append(function)
    lines = []
    if names:
        lines.append("- Functions: " + ", ".join(names[:10]))
    errors = [name for name, _ in oracle["errors"]]
    if errors:
        lines.append("- Exception types: " + ", ".join(errors))
    if with_commands:
        commands: list[str] = []
        for event in oracle["events"]:
            if event["kind"] == "test" and event.get("args") and event["args"] not in commands:
                commands.append(event["args"])
        if commands:
            lines.append("- Test commands: " + "; ".join(_clip(c, 80) for c in commands[:4]))
    return lines or ["- Python code with a pytest suite."]


def _pending_lines(oracle: dict[str, Any]) -> list[str]:
    events = oracle["events"]
    issue = oracle["issue"]
    targets = TEST_PATH.findall(issue)
    last_edit = max((i for i, e in enumerate(events) if e["kind"] == "edit"), default=-1)
    lines: list[str] = []
    for target in targets[:1]:
        runs = [e for e in events if e["kind"] == "test" and e.get("path") == target]
        if not runs or not runs[-1].get("passed"):
            state = "latest run fail" if runs else "not run yet"
            lines.append(f"- Make the tests in {target} pass ({state}).")
    if "regression test" in issue.lower() and not any(
        e["kind"] == "edit" and (e.get("path") or "").startswith("tests/") for e in events
    ):
        lines.append("- Add the regression test the user asked for.")
    whole = [i for i, e in enumerate(events) if e["kind"] == "test" and not e.get("path")]
    if not whole or whole[-1] < last_edit or not events[whole[-1]].get("passed"):
        lines.append("- Run the whole suite and confirm it passes.")
    for message in oracle["user_messages"][1:]:
        lines.append(f"- Keep to the user's later request: {_clip(message, 160)}")
    return lines or ["- Nothing left beyond a final check."]


def _current_lines(oracle: dict[str, Any], exception_label: str = "Most recent exception") -> list[str]:
    events = oracle["events"]
    lines: list[str] = []
    if events:
        last = events[-1]
        what = last.get("path") or _clip(_command(last), 90)
        state = ""
        if last["kind"] == "test":
            state = f": {'pass' if last.get('passed') else 'fail'}"
        lines.append(f"- Previous action: {last['kind']} {what}{state}")
    edited = oracle["edited"]
    lines.append(f"- Most recently edited file: {edited[-1]}" if edited else "- Most recently edited file: none yet")
    lines.append(f"- Changed an existing file: {'yes' if edited else 'no'}")
    if oracle["created"]:
        lines.append(f"- Created file: {oracle['created'][-1]}")
    tests = [e for e in events if e["kind"] == "test"]
    if tests:
        last_test = tests[-1]
        lines.append(
            f"- Latest test run: {'pass' if last_test.get('passed') else 'fail'} "
            f"({_clip(_command(last_test), 80)})"
        )
    if oracle["errors"]:
        lines.append(f"- {exception_label}: {oracle['errors'][-1][0]}")
    if oracle["frames"]:
        lines.append(f"- Most recent traceback ended in function: {oracle['frames'][-1][1]}")
    return lines


def _next_line(oracle: dict[str, Any]) -> str:
    # A bare "Next action: <kind> <file>" line: probe accuracy on next_step went from 0.46 to 0.83
    # against a version with a reason clause, whose verbs the probe agent echoed as the kind.
    following = oracle.get("next")
    if not following:
        return "Next action: other"
    path = following.get("path") or ""
    return f"Next action: {following['kind']}" + (f" {path}" if path else "")


def compose(oracle: dict[str, Any], label: dict[str, Any], level: int = 0, style: str = "compact") -> str:
    """The summary at a trimming level; higher levels drop more detail.

    `compact` (the default) lists files by role on a few lines, keeps tests and edits in Problem
    solving and counts the reads, quotes the issue once, and names the exception "in the output"
    (the probe confused it with exception types named in user messages). `full` is the first
    format: one line per file and every step.
    """
    keep = _fact_files(label)
    extra_files = [99, 99, 8, 4, 0, 0, 0, 0][min(level, 7)]
    full = style == "full"
    solving = _solving_lines(oracle) if full else _compact_solving_lines(oracle)
    solving_keep = [None, 24, 16, 12, 8, 6, MIN_SOLVING_LINES, MIN_SOLVING_LINES][min(level, 7)]
    if solving_keep is not None and len(solving) > solving_keep:
        dropped = len(solving) - solving_keep
        solving = [f"- ({dropped} earlier steps: reading and testing as listed above.)", *solving[-solving_keep:]]
    users = oracle["user_messages"]
    limit = MAX_USER_CHARS if level < 6 else 200
    issue = _clip(oracle["issue"], 600)
    first = _first_sentence(oracle["issue"])
    if full:
        primary = [f"Task: {first}.", f"The user asked: {issue}"]
        messages = [f"{i + 1}. {_clip(m, limit)}" for i, m in enumerate(users)] or ["1. (none)"]
        files = _file_lines(oracle, keep, extra_files)
    else:
        rest = issue[len(first) :].lstrip(". ").strip()
        primary = [f"Task: {first}."] + ([f"Details: {rest}"] if rest else [])
        messages = [f"1. {first} (the task above)."] + [f"{i + 2}. {_clip(m, limit)}" for i, m in enumerate(users[1:])]
        files = _compact_file_lines(oracle, keep, extra_files)
    body = {
        "Primary request and intent": primary,
        "Key technical concepts": _concept_lines(oracle, with_commands=full and level < 5),
        "Files and code sections": files,
        "Errors and fixes": _error_lines(oracle, with_messages=level < 7),
        "Problem solving": solving or ["- Nothing beyond reading so far."],
        "All user messages": messages,
        "Pending tasks": _pending_lines(oracle),
        "Current work": _current_lines(oracle, "Most recent exception" if full else "Most recent exception in the output"),
        "Next step": [_next_line(oracle)],
    }
    parts = []
    for heading in HEADINGS:
        parts.append(f"## {heading}\n" + "\n".join(body[heading]))
    return "\n\n".join(parts)


def target_tokens(label: dict[str, Any]) -> int:
    return int(min(CAP, SHARE * int(label["budget_tokens"])))


def grounded(oracle: dict[str, Any], session: str) -> dict[str, Any]:
    """The oracle with the next action's file dropped when the session has not shown it yet.

    The next action comes from after the cut; its file may be one the agent has not opened, and
    naming it would count as an invented identifier and teach the model to guess paths.
    """
    from microtensor_subnet.validator import compress

    following = oracle.get("next")
    if not following or not following.get("path"):
        return oracle
    if compress.mentioned(compress.normalise(session), following["path"], loose=True):
        return oracle
    return {**oracle, "next": {**following, "path": ""}}


def write_target(item: dict[str, Any], style: str = "") -> tuple[str, int, int]:
    """The summary for an item, its token count and the trimming level used."""
    style = style or os.environ.get("CCSLM_TARGET_STYLE", "compact")
    goal = target_tokens(item["label"])
    oracle = grounded(item["oracle"], item["task"]["user"])
    text = ""
    tokens = 0
    for level in range(8):
        text = compose(oracle, item["label"], level, style)
        tokens = count_tokens(text)
        if tokens <= goal:
            return text, tokens, level
    return text, tokens, 8


def verify(item: dict[str, Any], summary: str) -> dict[str, Any]:
    """The grader's deterministic parts on one summary, and the probe answers it states."""
    from microtensor_subnet.validator import compress

    label = item["label"]
    session = item["task"]["user"]
    tokens = count_tokens(summary)
    found = compress.sections(summary)
    report = {
        "valid": found is not None,
        "tokens": tokens,
        "budget": int(label["budget_tokens"]),
        "within_budget": tokens <= int(label["budget_tokens"]),
        "recall": compress.recall(summary.strip(), label.get("facts") or []),
        "invented": compress.hallucination(summary.strip(), session),
        "invented_names": sorted(
            name
            for name in compress.identifiers(summary)
            if not compress.mentioned(compress.normalise(session), name, loose=True)
        ),
    }
    report["deterministic_score"] = (1 - report["invented"]) * 0.5 * report["recall"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="write training pairs from oracle items")
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True, help="pairs jsonl: {item_id, prompt, completion}")
    parser.add_argument("--report", default="")
    parser.add_argument("--style", choices=["compact", "full"], default="")
    args = parser.parse_args()
    pairs = []
    bad: list[dict[str, Any]] = []
    levels: dict[int, int] = {}
    for item in iter_jsonl(args.items):
        summary, tokens, level = write_target(item, args.style)
        levels[level] = levels.get(level, 0) + 1
        check = verify(item, summary)
        if not (check["valid"] and check["within_budget"] and check["recall"] == 1.0 and check["invented"] == 0):
            bad.append({"item_id": item["item_id"], **check})
            continue
        pairs.append(
            {
                "item_id": item["item_id"],
                "family": item["family"],
                "prompt": slm_prompt(item["task"]),
                "completion": summary,
                "completion_tokens": tokens,
                "input_tokens": item["label"]["input_tokens"],
            }
        )
    written = write_jsonl(args.out, pairs)
    report = {"pairs": written, "rejected": len(bad), "levels": levels, "rejected_items": bad[:20]}
    write_json(args.report or Path(args.out).with_suffix(".report.json"), report)
    log(f"{written} pairs, {len(bad)} rejected, trimming levels {levels}")


if __name__ == "__main__":
    main()
