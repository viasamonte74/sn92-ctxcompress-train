"""Items in the arena's exact format, built with the arena's own session generator and label code.

Every item is what `mta dev` / `mta hidden` would emit (task, label, family, twin_of), plus an
`oracle` block: what happened in the session up to the cut, taken from the structured session the
renderer saw. `targets.py` writes the training summary from that block; nothing in it is used for
grading. Splits use disjoint generator seeds:

  train  dev-like lengths (12k to 95k tokens), the arena's two cuts plus extra cuts per session
  val    hidden-like (35k to 95k, full items at least 40k tokens), the arena's two cuts only
  test   hidden-like, a separate seed, 100 sessions by default like one SLM pool
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Any

from ccslm.common import log, write_json, write_jsonl

SPLITS: dict[str, dict[str, Any]] = {
    "train": {"tokens": (12_000, 95_000), "min_tokens": 12_000, "extra_cuts": 2},
    "val": {"tokens": (35_000, 95_000), "min_tokens": 40_000, "extra_cuts": 0},
    "test": {"tokens": (35_000, 95_000), "min_tokens": 40_000, "extra_cuts": 0},
}
EXTRA_SPAN = (0.25, 0.97)
MAX_NOTE = 240
MAX_ARGS = 200
SUMMARY_LINE = re.compile(r"^=*\s*((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?)[, ]*)+.*?)\s*=*$")


def _result_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        found = SUMMARY_LINE.match(line.strip())
        if found:
            return found.group(1).strip()
    return ""


def oracle_block(session: Any, cut: int, rendered: str) -> dict[str, Any]:
    """What the session shows up to `cut`, restricted to text that is visible in `rendered`."""
    from arenas.compress import facts as factsmod
    from arenas.compress import observe as obs

    def visible(value: str) -> bool:
        return bool(value) and factsmod.mentions(rendered, value)

    seen = obs.observe(session, cut)
    events: list[dict[str, Any]] = []
    users: list[str] = []
    pending: list[tuple[Any, str]] = []
    for step in session.steps[:cut]:
        if step.role == "user":
            users.append(step.text.strip())
        elif step.role == "agent":
            note = " ".join(step.text.split())[:MAX_NOTE]
            pending = [(call, note if i == 0 else "") for i, call in enumerate(step.calls)]
        elif step.role == "tool" and pending:
            call, note = pending.pop(0)
            path = session.relative(call.path) if call.path else ""
            event: dict[str, Any] = {
                "kind": call.kind,
                "name": call.name,
                "path": path if visible(path) else "",
                "args": call.args[:MAX_ARGS] if call.args and call.args[:60] in rendered else "",
                "note": note if note and note[:60] in rendered else "",
                "status": "ok" if step.ok else ("error" if step.ok is False else "done"),
            }
            if call.kind == "test":
                event["passed"] = obs._test_passed(step.text, step.ok)
                line = _result_line(step.text)
                event["result"] = line if line and line in rendered else ""
            errors: list[list[str]] = []
            for name, message in obs.ERROR_LINE.findall(step.text):
                message = (message or "").strip()[:120]
                if not visible(name) or any(e[0] == name for e in errors):
                    continue
                errors.append([name, message if message and message in rendered else ""])
            event["errors"] = errors[:3]
            events.append(event)
    following = obs.next_action(session, cut)
    errors_seen: list[list[str]] = []
    for name, message in seen.errors:
        if not visible(name):
            continue
        errors_seen = [e for e in errors_seen if e[0] != name]
        errors_seen.append([name, message if message and message in rendered else ""])
    frames = [[p, f] for p, f in seen.frames if f != "<module>" and visible(f)]
    return {
        "issue": users[0] if users else "",
        "user_messages": users,
        "events": events,
        "opened": [p for p in seen.opened if visible(p)],
        "created": [p for p in seen.created if visible(p)],
        "edited": [p for p in seen.edited if visible(p)],
        "errors": errors_seen,
        "frames": frames,
        "symbols": [s for s in seen.symbols if visible(s)],
        "tests": list(seen.tests),
        "next": {"kind": following[0], "path": following[1]} if following else None,
        "goal": obs.goal_of(session),
    }


def _extra_cuts(session: Any, rng: random.Random, taken: set[int], count: int) -> list[int]:
    from arenas.compress import build

    total = len(session.steps)
    bounds = [
        b
        for b in build._tool_boundaries(session)
        if EXTRA_SPAN[0] * total <= b <= EXTRA_SPAN[1] * total and b not in taken
    ]
    rng.shuffle(bounds)
    return sorted(bounds[:count])


def build_split(
    split: str,
    seed: str,
    sessions: int,
    out: Path,
    *,
    tokens: tuple[int, int] | None = None,
    min_tokens: int | None = None,
    extra_cuts: int | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    from arenas.compress import build, synth
    from arenas.compress.sources import parse
    from arenas.pool import Pool

    shape = SPLITS[split]
    span = tokens or shape["tokens"]
    floor = shape["min_tokens"] if min_tokens is None else min_tokens
    extra = shape["extra_cuts"] if extra_cuts is None else extra_cuts
    count, _ = build.counter()
    rng = random.Random(f"{seed}:cuts")
    pool = Pool(arena_id="ctx_compress", seed=0)
    skipped = {"unusable": 0, "no_cuts": 0, "thin": 0}
    made = 0
    started = time.time()
    for row in synth.rows(seed, sessions, span, workers=workers):
        session = parse(row, "synth")
        if not build.usable(session):
            skipped["unusable"] += 1
            continue
        cuts = build.choose_cuts(session, rng)
        if cuts is None:
            skipped["no_cuts"] += 1
            continue
        half, late = cuts
        stem = f"cc-{split}-{made:05d}"
        full = build.make_item(session, late, count, stem, ["slm"], min_tokens=floor)
        if full is None:
            skipped["thin"] += 1
            continue
        chosen = [(full, late)]
        twin = build.make_item(
            session, half, count, f"{stem}-twin", ["slm"], min_tokens=floor // 2, twin_of=stem
        )
        if twin is not None:
            chosen.append((twin, half))
        for k, cut in enumerate(_extra_cuts(session, rng, {half, late}, extra)):
            more = build.make_item(
                session, cut, count, f"{stem}-x{k}", ["slm"], min_tokens=floor // 2
            )
            if more is not None:
                more["extra_cut"] = True
                chosen.append((more, cut))
        for item, cut in chosen:
            item["oracle"] = oracle_block(session, cut, item["task"]["user"])
            pool.items.append(item)
        made += 1
        if made % 25 == 0:
            log(f"{split}: {made}/{sessions} sessions, {len(pool.items)} items, {time.time() - started:.0f}s")
    build.add_goal_probes(pool, rng)
    for item in pool.items:
        item["label"].pop("goal", None)
    written = write_jsonl(out / "items.jsonl", pool.items)
    report = {
        "split": split,
        "seed": seed,
        "sessions": made,
        "items": written,
        "twins": sum(1 for i in pool.items if i.get("twin_of")),
        "extra": sum(1 for i in pool.items if i.get("extra_cut")),
        "skipped": skipped,
        "input_tokens": build.stats([i["label"]["input_tokens"] for i in pool.items]),
        "budget_tokens": build.stats([i["label"]["budget_tokens"] for i in pool.items]),
        "seconds": round(time.time() - started, 1),
    }
    problems = build.check(pool)
    report["problems"] = problems[:20]
    write_json(out / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="build ctx_compress items with the arena generator")
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--seed", default="", help="generator seed string; default ccslm-<split>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", default="", help="session size span, e.g. 12000-95000")
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--extra-cuts", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    span = tuple(int(v) for v in args.tokens.split("-")) if args.tokens else None
    report = build_split(
        args.split,
        args.seed or f"ccslm-{args.split}",
        args.sessions,
        Path(args.out),
        tokens=span,  # type: ignore[arg-type]
        min_tokens=args.min_tokens,
        extra_cuts=args.extra_cuts,
        workers=args.workers,
    )
    log(f"{args.split}: {report['items']} items from {report['sessions']} sessions, {len(report['problems'])} problems")


if __name__ == "__main__":
    main()
