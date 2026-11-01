"""Run a model on items through a vLLM OpenAI server, the way the SLM pool calls it, with timings.

Each request is a raw completion of the arena's slm_prompt, greedy, max 3,072 tokens, stopped at
<|end_of_text|>, streamed so the first token time and the time per output token are measured on
this side. Start the server with scripts/serve_vllm.sh (batch invariant mode and the arena's
context length, like the official engine image). Timings are only meaningful at --concurrency 1,
which is how the envelope is measured; quality runs can use more.

  python -m ccslm.generate --items out/data/test/items.jsonl --out out/gen/base.jsonl --concurrency 4
  python -m ccslm.generate envelope --outputs out/gen/base.jsonl          # p95 figures and the gate
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ccslm.common import EXPECTED, iter_jsonl, log, slm_prompt, write_json

LONG_ABOVE = 32_768
LIMITS = {"ttft_ms": 4000.0, "tpot_ms": 20.0, "long_ttft_ms": 25000.0, "long_e2e_ms": 75000.0}
VIOLATION_SHARE = 0.01


def complete(url: str, prompt: str, *, model: str, max_tokens: int, seed: int, timeout: float) -> dict[str, Any]:
    import httpx

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stop": EXPECTED["stop"],
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text: list[str] = []
    first = 0.0
    usage: dict[str, Any] = {}
    finish = ""
    started = time.perf_counter()
    with httpx.Client(timeout=timeout) as http, http.stream("POST", url.rstrip("/") + "/v1/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = choice.get("text") or ""
                if piece and not first:
                    first = time.perf_counter()
                text.append(piece)
                finish = choice.get("finish_reason") or finish
    ended = time.perf_counter()
    completion = int(usage.get("completion_tokens") or 0)
    ttft = (first - started) * 1000 if first else (ended - started) * 1000
    tpot = (ended - first) * 1000 / (completion - 1) if first and completion > 1 else 0.0
    return {
        "text": "".join(text),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "ttft_ms": round(ttft, 1),
        "tpot_ms": round(tpot, 3),
        "latency_ms": round((ended - started) * 1000, 1),
        "finish_reason": finish,
    }


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The arena's per item envelope rule (score.envelope_share) and the p95 figures."""
    answered = [r for r in rows if r.get("completion_tokens")]
    over = 0
    for r in answered:
        long = r["prompt_tokens"] > LONG_ABOVE
        late = r["ttft_ms"] > (LIMITS["long_ttft_ms"] if long else LIMITS["ttft_ms"])
        late = late or (long and r["latency_ms"] > LIMITS["long_e2e_ms"])
        slow = r["completion_tokens"] > 1 and r["tpot_ms"] > LIMITS["tpot_ms"]
        over += 1 if late or slow else 0
    share = over / len(answered) if answered else 1.0
    short = [r for r in answered if r["prompt_tokens"] <= LONG_ABOVE]
    long_rows = [r for r in answered if r["prompt_tokens"] > LONG_ABOVE]
    return {
        "items": len(answered),
        "over": over,
        "violation_share": round(share, 4),
        "envelope_pass": share <= VIOLATION_SHARE,
        "ttft_p95_ms": p95([r["ttft_ms"] for r in answered]),
        "ttft_p95_short_ms": p95([r["ttft_ms"] for r in short]),
        "ttft_p95_long_ms": p95([r["ttft_ms"] for r in long_rows]),
        "e2e_p95_long_ms": p95([r["latency_ms"] for r in long_rows]),
        "tpot_p95_ms": p95([r["tpot_ms"] for r in answered]),
        "completion_tokens_mean": round(sum(r["completion_tokens"] for r in answered) / max(1, len(answered)), 1),
        "cut_off": sum(1 for r in answered if r.get("finish_reason") == "length"),
    }


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    done = {r["item_id"] for r in iter_jsonl(out)} if out.is_file() else set()
    items = [i for i in iter_jsonl(args.items) if i["item_id"] not in done and not i.get("extra_cut")]
    if args.limit:
        originals = [i for i in items if not i.get("twin_of")][: args.limit]
        keep = {i["item_id"] for i in originals}
        items = originals + [i for i in items if i.get("twin_of") in keep]
    if args.longest_first:
        items.sort(key=lambda i: -int(i["label"]["input_tokens"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    handle = out.open("a", encoding="utf-8")
    count = [0]

    def one(item: dict[str, Any]) -> None:
        try:
            result = complete(
                args.server,
                slm_prompt(item["task"]),
                model=args.model,
                max_tokens=args.max_tokens,
                seed=args.seed,
                timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - a failed call is recorded and counts as wrong
            result = {"text": "", "error": str(exc)[:300], "completion_tokens": 0, "prompt_tokens": 0}
        row = {"item_id": item["item_id"], "input_tokens": item["label"]["input_tokens"], **result}
        with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            count[0] += 1
            if count[0] % 10 == 0:
                log(f"generated {count[0]}/{len(items)}")

    log(f"{len(items)} items to generate ({len(done)} already done)")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        list(pool.map(one, items))
    handle.close()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "envelope":
        parser = argparse.ArgumentParser(description="envelope figures from a concurrency 1 generation run")
        parser.add_argument("cmd")
        parser.add_argument("--outputs", required=True)
        parser.add_argument("--out", default="")
        args = parser.parse_args()
        report = envelope(list(iter_jsonl(args.outputs)))
        if args.out:
            write_json(args.out, report)
        print(json.dumps(report, indent=2))
        return
    parser = argparse.ArgumentParser(description="generate summaries through a vLLM server")
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="slm")
    parser.add_argument("--max-tokens", type=int, default=EXPECTED["max_tokens"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="original items (twins follow)")
    parser.add_argument("--longest-first", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
