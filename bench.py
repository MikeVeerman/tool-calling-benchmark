#!/usr/bin/env python3
"""Local LLM tool-calling benchmark using Ollama + BitNet."""

import json
import re
import signal
import subprocess
import sys
import time

import ollama
import requests

# ---------------------------------------------------------------------------
# Tool definitions (Ollama format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Antwerp'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a pattern in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files, e.g. '*.py'",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with attendees at a given time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Meeting title",
                    },
                    "time": {
                        "type": "string",
                        "description": "Meeting time in ISO 8601 format",
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses",
                    },
                },
                "required": ["title", "time"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Fake tool implementations
# ---------------------------------------------------------------------------


def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 14, "condition": "Partly cloudy", "humidity": 72}


def search_files(pattern: str) -> dict:
    return {"pattern": pattern, "matches": ["src/main.py", "src/utils.py", "README.md"]}


def schedule_meeting(title: str, time: str, attendees: list | None = None) -> dict:
    return {"status": "scheduled", "title": title, "time": time, "attendees": attendees or []}


TOOL_DISPATCH = {
    "get_weather": get_weather,
    "search_files": search_files,
    "schedule_meeting": schedule_meeting,
}

# ---------------------------------------------------------------------------
# Test prompts – from obvious to ambiguous to trick to harder
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # P1 – obvious single-tool
    "What's the weather in Antwerp?",
    # P2 – obvious different tool
    "Find all Python files in the project.",
    # P3 – requires multiple args
    "Schedule a meeting called 'Sprint Review' for 2025-02-10T14:00:00 with alice@co.com and bob@co.com.",
    # P4 – ambiguous, could use tool or not
    "I'm heading to Brussels tomorrow, anything I should know?",
    # P5 – trick / meta question, should NOT call a tool
    "What tools do you have access to?",
    # P6 – multi-step reasoning (requires chaining context not available)
    "What's the weather in the city where we have our next sprint review?",
    # P7 – noisy parameter extraction
    "Oh hey, could you maybe like set up a meeting — 'Q3 Roadmap' — for next Tuesday at 3pm? I think dave@co.com and maybe susan@co.com should come",
    # P8 – adversarial: asks for TWO tools at once
    "Search for all files matching '*.py' and also tell me the weather in Paris.",
    # P9 – tool-adjacent trick, should NOT call a tool
    "Can you write a Python script that checks the weather using an API?",
]

# Indices of prompts where the correct behavior is to NOT call a tool
# P5 (idx 4): meta question about tools
# P9 (idx 8): asking to write code, not to call a tool
RESTRAINT_INDICES = {4, 8}

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

ALL_MODELS = [
    {"name": "qwen2.5:3b",    "backend": "ollama"},
    {"name": "qwen2.5:1.5b",  "backend": "ollama"},
    {"name": "qwen2.5:0.5b",  "backend": "ollama"},
    {"name": "llama3.2:3b",   "backend": "ollama"},
    {"name": "smollm2:1.7b",  "backend": "ollama"},
    {"name": "bitnet-3B",     "backend": "bitnet"},
]

# ---------------------------------------------------------------------------
# BitNet configuration
# ---------------------------------------------------------------------------

BITNET_DIR = "/home/mike/projects/bitnet"
BITNET_MODEL_PATH = f"{BITNET_DIR}/models/bitnet_b1_58-3B/ggml-model-i2_s.gguf"
BITNET_PORT = 8921

BITNET_SYSTEM_PROMPT = """\
You are a helpful assistant with access to the following tools. When the user's \
request can be fulfilled by calling a tool, respond with a tool call inside \
<tool_call></tool_call> tags. Otherwise, respond with plain text.

Available tools:

1. get_weather(city: string) – Get the current weather for a given city.
2. search_files(pattern: string) – Search for files matching a glob pattern.
3. schedule_meeting(title: string, time: string, attendees?: string[]) – Schedule a meeting.

To call a tool, respond EXACTLY like this (no other text before or after):
<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>

If the user's request does NOT require a tool call, just respond normally in plain text.
Do NOT call a tool if the user is asking you to write code, explain something, or answer a meta question.
"""

_bitnet_proc = None


def start_bitnet_server():
    """Start the BitNet llama-server as a subprocess."""
    global _bitnet_proc
    cmd = [
        f"{BITNET_DIR}/build/bin/llama-server",
        "-m", BITNET_MODEL_PATH,
        "--port", str(BITNET_PORT),
        "-c", "2048",
        "-np", "1",
    ]
    _bitnet_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Wait for server to be ready
    url = f"http://localhost:{BITNET_PORT}/health"
    for _ in range(60):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("BitNet server failed to start within 60s")


def stop_bitnet_server():
    """Stop the BitNet llama-server subprocess."""
    global _bitnet_proc
    if _bitnet_proc is not None:
        _bitnet_proc.terminate()
        try:
            _bitnet_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _bitnet_proc.kill()
            _bitnet_proc.wait()
        _bitnet_proc = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_one_ollama(model: str, prompt: str) -> dict:
    """Run a single prompt against an Ollama model and return result info."""
    messages = [{"role": "user", "content": prompt}]

    t0 = time.perf_counter()
    try:
        resp = ollama.chat(model=model, messages=messages, tools=TOOLS)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "tool_called": False,
            "tool_name": None,
            "valid_args": None,
            "latency_ms": round(elapsed),
            "error": str(e),
            "raw_content": None,
        }
    elapsed = (time.perf_counter() - t0) * 1000

    tool_calls = resp.message.tool_calls or []
    if not tool_calls:
        return {
            "tool_called": False,
            "tool_name": None,
            "valid_args": None,
            "latency_ms": round(elapsed),
            "error": None,
            "raw_content": resp.message.content,
        }

    tc = tool_calls[0]
    fname = tc.function.name
    args = tc.function.arguments

    try:
        json.dumps(args)
        valid = True
    except (TypeError, ValueError):
        valid = False

    return {
        "tool_called": True,
        "tool_name": fname,
        "valid_args": valid,
        "latency_ms": round(elapsed),
        "error": None,
        "raw_content": resp.message.content,
    }


def run_one_bitnet(prompt: str) -> dict:
    """Run a single prompt against the BitNet server and return result info."""
    url = f"http://localhost:{BITNET_PORT}/v1/chat/completions"
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
            "tool_called": False,
            "tool_name": None,
            "valid_args": None,
            "latency_ms": round(elapsed),
            "error": str(e),
            "raw_content": None,
        }
    elapsed = (time.perf_counter() - t0) * 1000

    content = data["choices"][0]["message"]["content"]

    # Parse <tool_call>...</tool_call> from response
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if not match:
        return {
            "tool_called": False,
            "tool_name": None,
            "valid_args": None,
            "latency_ms": round(elapsed),
            "error": None,
            "raw_content": content,
        }

    try:
        call = json.loads(match.group(1))
        fname = call.get("name", "")
        args = call.get("arguments", {})
        json.dumps(args)  # validate serialisable
        valid = True
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "tool_called": True,
            "tool_name": None,
            "valid_args": False,
            "latency_ms": round(elapsed),
            "error": None,
            "raw_content": content,
        }

    return {
        "tool_called": True,
        "tool_name": fname,
        "valid_args": valid,
        "latency_ms": round(elapsed),
        "error": None,
        "raw_content": content,
    }


def run_one(model_info: dict, prompt: str) -> dict:
    """Dispatch to the right backend."""
    if model_info["backend"] == "ollama":
        return run_one_ollama(model_info["name"], prompt)
    elif model_info["backend"] == "bitnet":
        return run_one_bitnet(prompt)
    else:
        raise ValueError(f"Unknown backend: {model_info['backend']}")


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def fmt_table(results: dict, model_list: list[dict]):
    """Print an ASCII summary table."""
    prompt_labels = [f"P{i+1}" for i in range(len(TEST_PROMPTS))]
    model_names = [m["name"] for m in model_list]

    print("\n" + "=" * 110)
    print("TEST PROMPTS")
    print("=" * 110)
    for i, p in enumerate(TEST_PROMPTS):
        restraint_tag = " [RESTRAINT]" if i in RESTRAINT_INDICES else ""
        print(f"  P{i+1}: {p}{restraint_tag}")

    print("\n" + "=" * 110)
    print("DETAILED RESULTS")
    print("=" * 110)

    hdr = f"{'Model':<18} {'Prompt':<6} {'Called?':<8} {'Tool':<20} {'Args OK':<8} {'ms':>6}"
    print(hdr)
    print("-" * len(hdr))

    for name in model_names:
        for i, r in enumerate(results[name]):
            called = "YES" if r["tool_called"] else "no"
            tool = r["tool_name"] or "-"
            args_ok = "OK" if r["valid_args"] else ("FAIL" if r["valid_args"] is False else "-")
            ms = str(r["latency_ms"])
            err = f"  ERR: {r['error']}" if r["error"] else ""
            print(f"{name:<18} {prompt_labels[i]:<6} {called:<8} {tool:<20} {args_ok:<8} {ms:>6}{err}")
        print("-" * len(hdr))

    # Summary table
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    shdr = f"{'Model':<18} {'Tool calls':<12} {'Valid args':<12} {'Avg ms':>8} {'Restraint':>10}"
    print(shdr)
    print("-" * len(shdr))

    for name in model_names:
        rs = results[name]
        n_called = sum(1 for r in rs if r["tool_called"])
        n_valid = sum(1 for r in rs if r["valid_args"])
        avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
        # Restraint: count of restraint prompts where model correctly did NOT call a tool
        restraint_pass = sum(
            1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"]
        )
        restraint_total = len(RESTRAINT_INDICES)
        print(
            f"{name:<18} {n_called:>3}/{len(rs):<8} {n_valid:>3}/{n_called if n_called else 0:<8} "
            f"{avg_ms:>7} {restraint_pass:>3}/{restraint_total}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    num_runs = 3
    model_names = [m["name"] for m in ALL_MODELS]

    print(f"Benchmarking {len(ALL_MODELS)} models x {len(TEST_PROMPTS)} prompts x {num_runs} runs")
    print(f"Tools available: {', '.join(t['function']['name'] for t in TOOLS)}")
    print()

    # Start BitNet server
    print("Starting BitNet server...")
    start_bitnet_server()
    print("BitNet server ready.\n")

    # all_runs[model_name][run_idx] = [result_per_prompt]
    all_runs = {m["name"]: [] for m in ALL_MODELS}
    # Store raw BitNet responses for logging
    bitnet_raw = {}  # (run_idx, prompt_idx) -> raw_content

    try:
        for run in range(num_runs):
            print(f"{'='*60}")
            print(f"  RUN {run + 1}/{num_runs}")
            print(f"{'='*60}")
            for model_info in ALL_MODELS:
                name = model_info["name"]
                print(f"--- {name} ---")
                run_results = []
                for i, prompt in enumerate(TEST_PROMPTS):
                    label = f"  P{i+1}"
                    print(f"{label}: {prompt[:60]}...", end=" ", flush=True)
                    r = run_one(model_info, prompt)
                    tag = r["tool_name"] or ("(no tool)" if not r["error"] else "ERROR")
                    print(f"=> {tag}  [{r['latency_ms']}ms]")
                    run_results.append(r)
                    # Capture BitNet raw output
                    if model_info["backend"] == "bitnet":
                        bitnet_raw[(run, i)] = r.get("raw_content")
                all_runs[name].append(run_results)
                print()
    finally:
        print("Stopping BitNet server...")
        stop_bitnet_server()
        print("BitNet server stopped.\n")

    # Build averaged results
    avg_results = {}
    for name in model_names:
        avg_results[name] = []
        for pi in range(len(TEST_PROMPTS)):
            entries = [all_runs[name][ri][pi] for ri in range(num_runs)]
            avg_lat = round(sum(e["latency_ms"] for e in entries) / num_runs)
            n_called = sum(1 for e in entries if e["tool_called"])
            called = n_called > num_runs / 2
            tool_names = [e["tool_name"] for e in entries if e["tool_name"]]
            tool_name = max(set(tool_names), key=tool_names.count) if tool_names else None
            n_valid = sum(1 for e in entries if e["valid_args"])
            valid = n_valid > 0 if called else None
            avg_results[name].append({
                "tool_called": called,
                "tool_name": tool_name if called else None,
                "valid_args": valid,
                "latency_ms": avg_lat,
                "error": None,
                "raw_content": None,
            })

    # Print per-run detail
    for run in range(num_runs):
        print(f"\n{'='*110}")
        print(f"RUN {run + 1} DETAILS")
        print("=" * 110)
        single = {m["name"]: all_runs[m["name"]][run] for m in ALL_MODELS}
        fmt_table(single, ALL_MODELS)

    # Print per-prompt latency breakdown
    print("\n" + "=" * 110)
    print(f"PER-PROMPT LATENCY ACROSS {num_runs} RUNS (ms)")
    print("=" * 110)
    plabels = [f"P{i+1}" for i in range(len(TEST_PROMPTS))]
    hdr = f"{'Model':<18} {'Prompt':<6} " + "  ".join(f"{'R'+str(r+1):>6}" for r in range(num_runs)) + f"  {'Avg':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name in model_names:
        for pi in range(len(TEST_PROMPTS)):
            vals = [all_runs[name][ri][pi]["latency_ms"] for ri in range(num_runs)]
            avg = round(sum(vals) / num_runs)
            cols = "  ".join(f"{v:>6}" for v in vals)
            print(f"{name:<18} {plabels[pi]:<6} {cols}  {avg:>6}")
        print("-" * len(hdr))

    # Print averaged summary
    print("\n" + "=" * 110)
    print(f"AVERAGED SUMMARY ({num_runs} runs)")
    print("=" * 110)
    fmt_table(avg_results, ALL_MODELS)

    # Print raw BitNet outputs for P1, P6, P8 (indices 0, 5, 7)
    print("\n" + "=" * 110)
    print("RAW BITNET OUTPUT (P1, P6, P8)")
    print("=" * 110)
    for pi, plabel in [(0, "P1"), (5, "P6"), (7, "P8")]:
        print(f"\n--- {plabel}: {TEST_PROMPTS[pi][:70]} ---")
        for run in range(num_runs):
            raw = bitnet_raw.get((run, pi), "(no output)")
            print(f"  Run {run+1}: {raw}")
        print()


if __name__ == "__main__":
    main()
