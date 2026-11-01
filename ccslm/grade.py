"""Grade summaries the way the SLM pool does, and report the pool's statistics.

Inputs are the items (with labels) and an outputs file of {item_id, text, ...timings}. The item
score is `microtensor_subnet.validator.compress.grade` with the Granite token counter and a probe
agent (see probe.py). The report gives the pass rate on original items, its Wilson lower bound
(z from the arena), the twin gap and factor, the paired lower bound against another graded run
(the base, or another variant), clustered by family, and the SLM score estimate
LB * twin * (1 + 0.25 * E) when efficiency figures are given.

  python -m ccslm.grade --items out/data/test/items.jsonl --outputs out/gen/base.jsonl --probe hf
  python -m ccslm.grade --items ... --oracle            # grade the training targets themselves
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ccslm.common import count_tokens, iter_jsonl, log, read_jsonl, write_json, write_jsonl

TWIN_TOLERANCE = 0.02
TWIN_SLOPE = 0.1
WILSON_Z = 1.96
EFFICIENCY_RANGE = 0.25
WEIGHTS = {"ttft": 0.4, "memory": 0.3, "parameters": 0.2, "tpot": 0.1}


class Memo:
    """Remembers the last answers so the per probe breakdown does not ask the probe twice."""

    def __init__(self, probe: Any) -> None:
        self.probe = probe
        self.key: tuple[str, tuple[str, ...]] | None = None
        self.value: list[str] = []

    def answer(self, summary: str, questions: Any) -> list[str]:
        key = (summary, tuple(questions))
        if key != self.key:
            self.key, self.value = key, list(self.probe.answer(summary, list(questions)))
        return list(self.value)


def judge(probe: Any, definition: dict[str, Any] | None = None) -> Any:
    from microtensor_subnet.validator import compress

    grading = (definition or {}).get("grading") or {}
    return compress.Judge(
        count_tokens=count_tokens,
        answer=Memo(probe).answer,
        pass_threshold=float(grading.get("pass_threshold") or 0.6),
        min_output_tokens=int(grading.get("min_output_tokens") or 1000),
        max_output_share=float(grading.get("max_output_share") or 0.05),
        max_output_tokens=int(grading.get("max_output_tokens") or 3072),
    )


def grade_one(item: dict[str, Any], text: str, the_judge: Any) -> dict[str, Any]:
    from microtensor_subnet.validator import compress

    label = item["label"]
    verdict = compress.grade(text, label, item["task"], the_judge)
    row: dict[str, Any] = {
        "item_id": item["item_id"],
        "family": item["family"],
        "twin_of": item.get("twin_of") or "",
        "score": verdict.score,
        "success": verdict.success,
        "reason": verdict.reason,
        "output_tokens": count_tokens(text),
        "budget": label.get("budget_tokens"),
    }
    if compress.sections(text) is not None and row["output_tokens"] <= int(label.get("budget_tokens") or 0):
        probes = label.get("probes") or []
        answers = the_judge.answer(text.strip(), [str(p.get("question", "")) for p in probes])
        row["probes"] = [
            {"type": p.get("type"), "q": str(p.get("question", ""))[:60], "a": a[:80], "s": compress.probe_score(p, a)}
            for p, a in zip(probes, answers, strict=False)
        ]
    return row


def wilson_lb(successes: int, n: int, z: float = WILSON_Z) -> float:
    from microtensor_subnet.scoring.stats import wilson_lb as lb

    return lb(successes, n, z)


def twin_factor(pass_rate: float, twin_rate: float) -> tuple[float, float]:
    gap = max(0.0, (pass_rate - twin_rate) - TWIN_TOLERANCE)
    return gap, 1 - min(1.0, gap / TWIN_SLOPE)


def efficiency(mine: dict[str, float], reference: dict[str, float]) -> float:
    total = 0.0
    for key, weight in WEIGHTS.items():
        ref, got = float(reference.get(key) or 0), float(mine.get(key) or 0)
        if ref > 0 and got > 0:
            total += weight * min(1.0, max(0.0, (ref - got) / ref))
    return total


def summarise(rows: list[dict[str, Any]], compare: list[dict[str, Any]] | None = None, z: float = WILSON_Z) -> dict[str, Any]:
    from microtensor_subnet.scoring.stats import paired_lb

    originals = [r for r in rows if not r["twin_of"]]
    twins = [r for r in rows if r["twin_of"]]
    passed = sum(1 for r in originals if r["success"])
    pass_rate = passed / len(originals) if originals else 0.0
    twin_rate = sum(1 for r in twins if r["success"]) / len(twins) if twins else 0.0
    gap, factor = twin_factor(pass_rate, twin_rate) if twins else (0.0, 1.0)
    parsed = [r for r in rows if r["reason"].startswith("recall")]
    metrics: dict[str, list[float]] = {"recall": [], "invented": [], "continuation": []}
    for r in parsed:
        for part in r["reason"].split(", "):
            name, _, value = part.partition(" ")
            if name in metrics:
                metrics[name].append(float(value))
    probe_types: dict[str, list[float]] = {}
    for r in rows:
        for p in r.get("probes") or []:
            probe_types.setdefault(str(p["type"]), []).append(float(p["s"]))
    failures = Counter(
        "format" if r["reason"].startswith("the nine headings")
        else "budget" if "output tokens, over" in r["reason"]
        else "score"
        for r in rows
        if not r["success"]
    )
    report: dict[str, Any] = {
        "n": len(originals),
        "twins": len(twins),
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "lower_bound": round(wilson_lb(passed, len(originals), z), 4),
        "twin_rate": round(twin_rate, 4),
        "twin_gap": round(gap, 4),
        "twin_factor": round(factor, 4),
        "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else 0.0,
        "mean": {k: round(sum(v) / len(v), 4) for k, v in metrics.items() if v},
        "probe_accuracy": {k: round(sum(v) / len(v), 3) for k, v in sorted(probe_types.items())},
        "failures": dict(failures),
        "output_tokens_mean": round(sum(r["output_tokens"] for r in rows) / len(rows), 1) if rows else 0,
    }
    if compare:
        theirs = {r["item_id"]: r for r in compare}
        mine = [r for r in originals if r["item_id"] in theirs]
        if mine:
            paired = paired_lb(
                [1.0 if r["success"] else 0.0 for r in mine],
                [1.0 if theirs[r["item_id"]]["success"] else 0.0 for r in mine],
                [r["family"] for r in mine],
            )
            report["paired"] = {
                "delta": paired.delta,
                "lower_bound": paired.lower_bound,
                "n": paired.n,
                "families": paired.families,
            }
    return report


def estimate(report: dict[str, Any], mine: dict[str, float] | None, reference: dict[str, float] | None) -> dict[str, Any]:
    e = efficiency(mine or {}, reference or {}) if mine and reference else 0.0
    return {
        "efficiency": round(e, 4),
        "slm_score": round(report["lower_bound"] * report["twin_factor"] * (1 + EFFICIENCY_RANGE * e), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="grade ctx_compress summaries like the SLM pool")
    parser.add_argument("--items", required=True)
    parser.add_argument("--outputs", default="", help="jsonl of {item_id, text}")
    parser.add_argument("--oracle", action="store_true", help="grade the training targets instead")
    parser.add_argument("--probe", default="hf", choices=["hf", "openai"])
    parser.add_argument("--probe-url", default="")
    parser.add_argument("--compare", default="", help="another graded.jsonl for the paired lower bound")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True, help="folder for graded.jsonl and report.json")
    args = parser.parse_args()

    from ccslm import probe as probe_module

    items = {i["item_id"]: i for i in iter_jsonl(args.items)}
    if args.oracle:
        from ccslm.targets import write_target

        outputs = {k: write_target(v)[0] for k, v in items.items()}
    else:
        outputs = {r["item_id"]: str(r.get("text") or "") for r in iter_jsonl(args.outputs)}
    chosen = [k for k in items if k in outputs]
    if args.limit:
        keep = [k for k in chosen if not items[k].get("twin_of")][: args.limit]
        chosen = keep + [k for k in chosen if items[k].get("twin_of") in set(keep)]
    missing = [k for k in items if k not in outputs and not items[k].get("extra_cut")]
    the_judge = judge(probe_module.make(args.probe, args.probe_url))
    rows = []
    for n, key in enumerate(chosen, 1):
        rows.append(grade_one(items[key], outputs[key], the_judge))
        if n % 10 == 0:
            log(f"graded {n}/{len(chosen)}")
    out = Path(args.out)
    write_jsonl(out / "graded.jsonl", rows)
    report = summarise(rows, read_jsonl(args.compare) if args.compare else None)
    report["missing_outputs"] = len(missing)
    write_json(out / "report.json", report)
    log(
        f"pass {report['passed']}/{report['n']} = {report['pass_rate']}, LB {report['lower_bound']}, "
        f"twin {report['twin_rate']} (factor {report['twin_factor']}), mean {report['mean']}"
    )


if __name__ == "__main__":
    main()
