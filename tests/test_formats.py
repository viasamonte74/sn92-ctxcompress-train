"""The quantised weight format check, the envelope rule and the streaming client, without a GPU."""

from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from ccslm import generate, quantize
from ccslm.common import portable_config


def header_file(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = 4
        for d in shape:
            size *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_gptq_layout_passes(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False}}))
    stem = "model.layers.5.self_attn.q_proj"
    header_file(
        tmp_path / "model.safetensors",
        {f"{stem}.qweight": ("I32", [16, 64]), f"{stem}.qzeros": ("I32", [1, 8]), f"{stem}.scales": ("F16", [1, 64]), f"{stem}.g_idx": ("I32", [128])},
    )
    assert quantize.check_format(tmp_path) == []


def test_compressed_tensors_layout_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "compressed-tensors", "bits": 4, "group_size": 128}}))
    header_file(tmp_path / "model.safetensors", {"model.layers.0.mlp.weight_packed": ("I32", [8, 8]), "model.layers.0.mlp.weight_scale": ("F16", [8, 1])})
    problems = quantize.check_format(tmp_path)
    assert any("quant_method" in p for p in problems)
    assert any("cannot read" in p for p in problems)
    assert any("not quantised" in p for p in problems)


def test_portable_config_restores_the_base_naming() -> None:
    reference = {"layer_types": ["mamba", "attention", "mamba"], "num_hidden_layers": 3, "torch_dtype": "bfloat16", "time_step_limit": None}
    saved = {"layer_types": ["linear_attention", "full_attention"], "dtype": "bfloat16", "time_step_limit": [0.0, {"__float__": "Infinity"}]}
    out = portable_config(saved, reference)
    assert out["layer_types"] == ["mamba", "attention"] and out["num_hidden_layers"] == 2
    assert out["time_step_limit"] is None and "dtype" not in out


def test_envelope_rule() -> None:
    rows = [{"prompt_tokens": 20_000, "completion_tokens": 900, "ttft_ms": 3000, "tpot_ms": 9, "latency_ms": 11000}] * 99
    late = {"prompt_tokens": 20_000, "completion_tokens": 900, "ttft_ms": 4500, "tpot_ms": 9, "latency_ms": 12000}
    report = generate.envelope([*rows, late])
    assert report["over"] == 1 and report["envelope_pass"]
    long_slow = {"prompt_tokens": 90_000, "completion_tokens": 3000, "ttft_ms": 20_000, "tpot_ms": 21, "latency_ms": 80_000}
    assert not generate.envelope([*rows, late, long_slow])["envelope_pass"]


class Stub(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        assert body["temperature"] == 0 and body["stop"] == ["<|end_of_text|>"] and body["stream"]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for piece in ("## Primary", " request and intent\n", "x"):
            self.wfile.write(f"data: {json.dumps({'choices': [{'text': piece, 'finish_reason': None}]})}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 3}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None:
        pass


def test_streaming_client_reads_text_usage_and_timings() -> None:
    server = HTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        result = generate.complete(url, "prompt", model="slm", max_tokens=16, seed=0, timeout=10)
    finally:
        server.shutdown()
    assert result["text"] == "## Primary request and intent\nx"
    assert result["prompt_tokens"] == 10 and result["completion_tokens"] == 3
    assert result["finish_reason"] == "stop" and result["ttft_ms"] >= 0
