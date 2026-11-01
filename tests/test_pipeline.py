"""A dry run of the orchestrator with a fake `vllm` on PATH: serve, generate, grade, envelope, report."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAKE_VLLM = r'''#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
SUMMARY = "## Primary request and intent\nFix it.\n\n" + "\n\n".join("## " + h + "\n-" for h in [
    "Key technical concepts", "Files and code sections", "Errors and fixes", "Problem solving",
    "All user messages", "Pending tasks", "Current work", "Next step"])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{{"data": []}}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.send_response(200)
        if self.path.endswith("/chat/completions"):
            self.end_headers()
            self.wfile.write(json.dumps({{"choices": [{{"message": {{"content": "unknown"}}}}]}}).encode())
            return
        self.send_header("content-type", "text/event-stream"); self.end_headers()
        for piece in (SUMMARY[:20], SUMMARY[20:]):
            self.wfile.write(("data: " + json.dumps({{"choices": [{{"text": piece}}]}}) + "\n\n").encode())
        usage = {{"prompt_tokens": 40000, "completion_tokens": 60}}
        self.wfile.write(("data: " + json.dumps({{"choices": [{{"text": "", "finish_reason": "stop"}}], "usage": usage}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pipeline_dry_run(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "vllm"
    fake.write_text(FAKE_VLLM.format(python=sys.executable))
    fake.chmod(0o755)
    base = tmp_path / "base"
    base.mkdir()
    port = free_port()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CCSLM_OUT": str(tmp_path / "out"),
        "CCSLM_BASE_DIR": str(base),
        "CCSLM_SPLIT_SESSIONS": "test=2,val=1,train=1",
        "CCSLM_VLLM_PORT": str(port),
        "CCSLM_PROBE_URL": f"http://127.0.0.1:{port}",
    }

    def stages(names: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ccslm.pipeline", "--stages", names], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900
        )

    done = stages("data,targets")
    assert done.returncode == 0, done.stderr[-3000:]
    done = stages("baseline")
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads((tmp_path / "out" / "eval" / "base" / "report.json").read_text())
    assert report["n"] == 2 and report["failures"] == {"score": report["n"] + report["twins"]}
    again = stages("baseline")
    assert "already done" in again.stderr
    gen = [json.loads(line) for line in (tmp_path / "out" / "gen" / "base.jsonl").read_text().splitlines()]
    assert all(row["text"].startswith("## Primary request and intent") for row in gen)
