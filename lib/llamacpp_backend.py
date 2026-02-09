"""llama.cpp backend: server lifecycle and runner for GGUF models not supported by Ollama."""

import subprocess
import time

import requests

from lib.bitnet_backend import (
    BITNET_SYSTEM_PROMPT,
    _parse_tool_call_from_text,
    _parse_all_tool_calls_from_text,
)

# ---------------------------------------------------------------------------
# llama.cpp configuration
# ---------------------------------------------------------------------------

LLAMACPP_SERVER = "/home/mike/projects/llama.cpp/build/bin/llama-server"
LLAMACPP_PORT = 8922

_llamacpp_proc = None
_llamacpp_current_model = None


def start_llamacpp_server(model_id: str):
    """Start llama-server as a subprocess, downloading from HF if needed."""
    global _llamacpp_proc, _llamacpp_current_model
    if _llamacpp_proc is not None and _llamacpp_current_model == model_id:
        return
    if _llamacpp_proc is not None:
        stop_llamacpp_server()
    cmd = [
        LLAMACPP_SERVER,
        "-hf", model_id,
        "--port", str(LLAMACPP_PORT),
        "-c", "4096",
        "-np", "1",
    ]
    _llamacpp_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _llamacpp_current_model = model_id
    # Wait for server to be ready
    url = f"http://localhost:{LLAMACPP_PORT}/health"
    for _ in range(120):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"llama-server failed to start within 120s for {model_id}")


def stop_llamacpp_server():
    """Stop the llama-server subprocess."""
    global _llamacpp_proc, _llamacpp_current_model
    if _llamacpp_proc is not None:
        _llamacpp_proc.terminate()
        try:
            _llamacpp_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llamacpp_proc.kill()
            _llamacpp_proc.wait()
        _llamacpp_proc = None
        _llamacpp_current_model = None


def run_one_llamacpp(model_id: str, prompt: str) -> dict:
    """Run a single prompt against the llama-server and return result info."""
    url = f"http://localhost:{LLAMACPP_PORT}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": BITNET_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "tool_called": False, "tool_name": None, "valid_args": None,
            "latency_ms": round(elapsed), "error": str(e), "raw_content": None,
            "all_tool_calls": [],
        }
    elapsed = (time.perf_counter() - t0) * 1000

    content = data["choices"][0]["message"]["content"]

    all_parsed = _parse_all_tool_calls_from_text(content)
    parsed = _parse_tool_call_from_text(content)
    if not parsed:
        return {
            "tool_called": False, "tool_name": None, "valid_args": None,
            "latency_ms": round(elapsed), "error": None, "raw_content": content,
            "all_tool_calls": all_parsed,
        }
    if not parsed["valid"]:
        return {
            "tool_called": True, "tool_name": None, "valid_args": False,
            "latency_ms": round(elapsed), "error": None, "raw_content": content,
            "all_tool_calls": all_parsed,
        }
    return {
        "tool_called": True, "tool_name": parsed["name"], "valid_args": True,
        "latency_ms": round(elapsed), "error": None, "raw_content": content,
        "all_tool_calls": all_parsed,
    }
