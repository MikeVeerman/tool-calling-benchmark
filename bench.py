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

# Indices of prompts where calling a valid tool is clearly correct (for Agent Score)
# P1, P2, P3, P6, P7 are clear tool-call prompts; P4 is ambiguous, P8 is dual-tool
TOOL_CALL_INDICES = {0, 1, 2, 5, 6, 3, 7}  # 7 prompts

# ---------------------------------------------------------------------------
# Backend display names
# ---------------------------------------------------------------------------

BACKEND_DISPLAY = {
    "ollama":     ("Ollama",     "native-tools"),
    "ollama_raw": ("Ollama",     "raw-schema"),
    "bitnet":     ("bitnet.cpp", "openai-compat"),
}

P8_REQUIRED_TOOLS = {"search_files", "get_weather"}


def get_backend_display(model_info: dict) -> tuple[str, str]:
    """Return (backend_name, mode) for display purposes."""
    return BACKEND_DISPLAY[model_info["backend"]]


# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

ALL_MODELS = [
    {"name": "qwen2.5:3b",      "backend": "ollama",  "origin": "CN"},
    {"name": "qwen2.5:1.5b",    "backend": "ollama",  "origin": "CN"},
    {"name": "qwen2.5:0.5b",    "backend": "ollama",  "origin": "CN"},
    {"name": "llama3.2:3b",     "backend": "ollama",  "origin": "US"},
    {"name": "smollm2:1.7b",    "backend": "ollama",  "origin": "US"},
    {"name": "ministral-3:3b",  "backend": "ollama",  "origin": "FR"},
    {"name": "deepseek-r1:1.5b","backend": "ollama_raw",  "origin": "CN"},
    {"name": "gemma3:1b",       "backend": "ollama_raw",  "origin": "US"},
    {"name": "phi4-mini:3.8b",  "backend": "ollama_raw",  "origin": "US"},
    {"name": "bitnet-3B",       "backend": "bitnet",  "origin": "US/1bit",
     "model_path": "/home/mike/projects/bitnet/models/bitnet_b1_58-3B/ggml-model-i2_s.gguf"},
    {"name": "bitnet-2B-4T",    "backend": "bitnet",  "origin": "US/1bit",
     "model_path": "/home/mike/projects/bitnet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"},
]

# Sub-2B models for the "edge agent" mini leaderboard
EDGE_MODELS = {"qwen2.5:0.5b", "qwen2.5:1.5b", "smollm2:1.7b", "deepseek-r1:1.5b", "gemma3:1b", "bitnet-2B-4T"}

# ---------------------------------------------------------------------------
# BitNet configuration
# ---------------------------------------------------------------------------

BITNET_DIR = "/home/mike/projects/bitnet"
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
_bitnet_current_model = None


def start_bitnet_server(model_path: str):
    """Start the BitNet llama-server as a subprocess for a given model."""
    global _bitnet_proc, _bitnet_current_model
    if _bitnet_proc is not None and _bitnet_current_model == model_path:
        return  # Already running the right model
    if _bitnet_proc is not None:
        stop_bitnet_server()
    cmd = [
        f"{BITNET_DIR}/build/bin/llama-server",
        "-m", model_path,
        "--port", str(BITNET_PORT),
        "-c", "2048",
        "-np", "1",
    ]
    _bitnet_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _bitnet_current_model = model_path
    # Wait for server to be ready
    url = f"http://localhost:{BITNET_PORT}/health"
    for _ in range(90):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"BitNet server failed to start within 90s for {model_path}")


def stop_bitnet_server():
    """Stop the BitNet llama-server subprocess."""
    global _bitnet_proc, _bitnet_current_model
    if _bitnet_proc is not None:
        _bitnet_proc.terminate()
        try:
            _bitnet_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _bitnet_proc.kill()
            _bitnet_proc.wait()
        _bitnet_proc = None
        _bitnet_current_model = None


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
            "all_tool_calls": [],
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
            "all_tool_calls": [],
        }

    # Build list of all tool calls
    all_tc = []
    for tc in tool_calls:
        fname = tc.function.name
        args = tc.function.arguments
        try:
            json.dumps(args)
            valid = True
        except (TypeError, ValueError):
            valid = False
        all_tc.append({"name": fname, "arguments": args, "valid": valid})

    # First call populates the existing top-level fields
    first = all_tc[0]
    return {
        "tool_called": True,
        "tool_name": first["name"],
        "valid_args": first["valid"],
        "latency_ms": round(elapsed),
        "error": None,
        "raw_content": resp.message.content,
        "all_tool_calls": all_tc,
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


def _parse_tool_call_from_text(content: str) -> dict | None:
    """Parse <tool_call>{...} from raw text using brace-counting for nested JSON."""
    idx = content.find("<tool_call>")
    if idx == -1:
        return None
    rest = content[idx + len("<tool_call>"):].lstrip()
    if not rest.startswith("{"):
        return None
    # Count braces to find the complete JSON object
    depth = 0
    end = -1
    for i, c in enumerate(rest):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        call = json.loads(rest[:end])
        fname = call.get("name", "")
        args = call.get("arguments", {})
        json.dumps(args)  # validate serialisable
        return {"name": fname, "arguments": args, "valid": True}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"name": None, "arguments": None, "valid": False}


def _parse_all_tool_calls_from_text(content: str) -> list[dict]:
    """Parse ALL <tool_call> blocks from raw text. Returns list of parsed dicts.

    Handles sequential <tool_call> blocks with or without closing </tool_call> tags,
    newline-separated JSON calls, and skips invalid JSON blocks.
    """
    results = []
    search_start = 0
    while True:
        idx = content.find("<tool_call>", search_start)
        if idx == -1:
            break
        rest = content[idx + len("<tool_call>"):].lstrip()
        if not rest.startswith("{"):
            search_start = idx + len("<tool_call>")
            continue
        # Count braces to find the complete JSON object
        depth = 0
        end = -1
        for i, c in enumerate(rest):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            search_start = idx + len("<tool_call>")
            continue
        try:
            call = json.loads(rest[:end])
            fname = call.get("name", "")
            args = call.get("arguments", {})
            json.dumps(args)  # validate serialisable
            results.append({"name": fname, "arguments": args, "valid": True})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # skip invalid block, continue to next
        # Advance past this block
        search_start = idx + len("<tool_call>") + end
    return results


def run_one_ollama_raw(model: str, prompt: str) -> dict:
    """Run a prompt via Ollama WITHOUT native tool API — use system prompt and parse text."""
    messages = [
        {"role": "system", "content": BITNET_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    t0 = time.perf_counter()
    try:
        resp = ollama.chat(model=model, messages=messages)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "tool_called": False, "tool_name": None, "valid_args": None,
            "latency_ms": round(elapsed), "error": str(e), "raw_content": None,
            "all_tool_calls": [],
        }
    elapsed = (time.perf_counter() - t0) * 1000

    content = resp.message.content or ""
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


def run_one(model_info: dict, prompt: str) -> dict:
    """Dispatch to the right backend."""
    if model_info["backend"] == "ollama":
        return run_one_ollama(model_info["name"], prompt)
    elif model_info["backend"] == "ollama_raw":
        return run_one_ollama_raw(model_info["name"], prompt)
    elif model_info["backend"] == "bitnet":
        return run_one_bitnet(prompt)
    else:
        raise ValueError(f"Unknown backend: {model_info['backend']}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_action_score(results_for_model: list[dict]) -> float:
    """Action Score = correct_tool_calls / 7 (actionable prompts P1-P4, P6-P8)."""
    rs = results_for_model
    n_valid_tool = sum(1 for idx in TOOL_CALL_INDICES if rs[idx]["valid_args"])
    return round(n_valid_tool / len(TOOL_CALL_INDICES), 3)


def compute_restraint_score(results_for_model: list[dict]) -> float:
    """Restraint Score = correct_refusals / 2 (restraint prompts P5, P9)."""
    rs = results_for_model
    restraint_pass = sum(1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"])
    return round(restraint_pass / len(RESTRAINT_INDICES), 3)


def compute_agent_score(results_for_model: list[dict]) -> float:
    """Agent Score = Action * 0.5 + Restraint * 0.5 (derived composite).

    Uses raw (unrounded) action and restraint to avoid double-rounding.
    """
    rs = results_for_model
    n_valid_tool = sum(1 for idx in TOOL_CALL_INDICES if rs[idx]["valid_args"])
    restraint_pass = sum(1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"])
    accuracy = n_valid_tool / len(TOOL_CALL_INDICES)
    restraint = restraint_pass / len(RESTRAINT_INDICES)
    return round(accuracy * 0.5 + restraint * 0.5, 3)


def compute_reliability(all_runs_for_model: list[list[dict]], num_runs: int) -> float:
    """Reliability = average per-prompt (successful_runs / total_runs).

    "Successful" means valid_args for actionable prompts, not tool_called for restraint.
    Requires per-run data (not majority-voted).
    """
    num_prompts = len(all_runs_for_model[0]) if all_runs_for_model else 0
    prompt_reliabilities = []
    for pi in range(num_prompts):
        successes = 0
        for ri in range(num_runs):
            r = all_runs_for_model[ri][pi]
            if pi in RESTRAINT_INDICES:
                if not r["tool_called"]:
                    successes += 1
            else:  # actionable prompt
                if r["valid_args"]:
                    successes += 1
        prompt_reliabilities.append(successes / num_runs)
    return round(sum(prompt_reliabilities) / num_prompts, 3) if prompt_reliabilities else 0.0


def compute_multi_tool_accuracy(results_for_model: list[dict], model_info: dict) -> float | None:
    """Multi-Tool Accuracy for P8 (index 7): len(called_tools & P8_REQUIRED_TOOLS) / 2.

    Returns None for backend == "ollama" (native API only captures first call).
    """
    if model_info["backend"] == "ollama":
        return None  # native API returns only first tool call
    p8 = results_for_model[7]  # P8 is index 7
    all_tc = p8.get("all_tool_calls", [])
    if not all_tc:
        # Fall back: if we have a single tool_name, use that
        called_tools = {p8["tool_name"]} if p8.get("tool_name") else set()
    else:
        called_tools = {tc["name"] for tc in all_tc if tc.get("valid")}
    return round(len(called_tools & P8_REQUIRED_TOOLS) / len(P8_REQUIRED_TOOLS), 3)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

# Lookup for origin by model name
_ORIGIN_MAP = {m["name"]: m["origin"] for m in ALL_MODELS}


def fmt_table(results: dict, model_list: list[dict], scores: dict | None = None):
    """Print an ASCII summary table. If scores dict provided, show extended columns."""
    prompt_labels = [f"P{i+1}" for i in range(len(TEST_PROMPTS))]
    model_names = [m["name"] for m in model_list]

    print("\n" + "=" * 160)
    print("TEST PROMPTS")
    print("=" * 160)
    for i, p in enumerate(TEST_PROMPTS):
        restraint_tag = " [RESTRAINT]" if i in RESTRAINT_INDICES else ""
        print(f"  P{i+1}: {p}{restraint_tag}")

    print("\n" + "=" * 160)
    print("DETAILED RESULTS")
    print("=" * 160)

    hdr = f"{'Model':<20} {'Prompt':<6} {'Called?':<8} {'Tool':<20} {'Args OK':<8} {'ms':>6}"
    print(hdr)
    print("-" * len(hdr))

    for name in model_names:
        for i, r in enumerate(results[name]):
            called = "YES" if r["tool_called"] else "no"
            tool = r["tool_name"] or "-"
            args_ok = "OK" if r["valid_args"] else ("FAIL" if r["valid_args"] is False else "-")
            ms = str(r["latency_ms"])
            err = f"  ERR: {r['error']}" if r["error"] else ""
            print(f"{name:<20} {prompt_labels[i]:<6} {called:<8} {tool:<20} {args_ok:<8} {ms:>6}{err}")
        print("-" * len(hdr))

    # Summary table sorted by Agent Score
    print("\n" + "=" * 160)
    print("SUMMARY (sorted by Agent Score)")
    print("=" * 160)

    if scores:
        shdr = (f"{'Model':<20} {'Backend':<12} {'Mode':<14} {'Origin':<9} "
                f"{'Action':>7} {'Restraint':>10} {'Reliability':>12} {'Multi-Tool':>11} "
                f"{'Agent Score':>12} {'Avg ms':>8}")
        print(shdr)
        print("-" * len(shdr))

        rows = []
        for name in model_names:
            rs = results[name]
            avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
            s = scores[name]
            rows.append((name, s["backend"], s["mode"], _ORIGIN_MAP.get(name, "??"),
                         s["action"], s["restraint"], s["reliability"],
                         s["multi_tool"], s["agent_score"], avg_ms))

        rows.sort(key=lambda r: r[8], reverse=True)

        for (name, backend, mode, origin, action, restraint, reliability,
             multi_tool, agent_score, avg_ms) in rows:
            mt_str = f"{multi_tool:.3f}" if multi_tool is not None else "N/A*"
            print(
                f"{name:<20} {backend:<12} {mode:<14} {origin:<9} "
                f"{action:>7.3f} {restraint:>10.3f} {reliability:>12.3f} {mt_str:>11} "
                f"{agent_score:>12.3f} {avg_ms:>7}"
            )
    else:
        shdr = (f"{'Model':<20} {'Origin':<9} {'Tool calls':<12} {'Valid args':<12} "
                f"{'Avg ms':>8} {'Restraint':>10} {'Agent Score':>12}")
        print(shdr)
        print("-" * len(shdr))

        rows = []
        for name in model_names:
            rs = results[name]
            n_called = sum(1 for r in rs if r["tool_called"])
            n_valid = sum(1 for r in rs if r["valid_args"])
            avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
            restraint_pass = sum(1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"])
            restraint_total = len(RESTRAINT_INDICES)
            score = compute_agent_score(rs)
            origin = _ORIGIN_MAP.get(name, "??")
            rows.append((name, origin, n_called, len(rs), n_valid, n_called, avg_ms,
                          restraint_pass, restraint_total, score))

        rows.sort(key=lambda r: r[9], reverse=True)

        for (name, origin, n_called, total, n_valid, n_called_denom,
             avg_ms, rpass, rtotal, score) in rows:
            print(
                f"{name:<20} {origin:<9} {n_called:>3}/{total:<8} "
                f"{n_valid:>3}/{n_called_denom if n_called_denom else 0:<8} "
                f"{avg_ms:>7} {rpass:>3}/{rtotal}      {score:>8.3f}"
            )

    print()


def fmt_edge_leaderboard(results: dict, model_list: list[dict], scores: dict | None = None):
    """Print a mini leaderboard of sub-2B 'edge agent' models."""
    edge_models = [m for m in model_list if m["name"] in EDGE_MODELS]
    if not edge_models:
        return

    print("\n" + "=" * 160)
    print("EDGE AGENT MINI LEADERBOARD (sub-2B models)")
    print("=" * 160)

    if scores:
        shdr = (f"{'#':<4} {'Model':<20} {'Backend':<12} {'Mode':<14} {'Origin':<9} "
                f"{'Action':>7} {'Restraint':>10} {'Reliability':>12} {'Multi-Tool':>11} "
                f"{'Agent Score':>12} {'Avg ms':>8}")
        print(shdr)
        print("-" * len(shdr))

        rows = []
        for m in edge_models:
            name = m["name"]
            rs = results[name]
            avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
            s = scores[name]
            rows.append((name, s["backend"], s["mode"], _ORIGIN_MAP.get(name, "??"),
                         s["action"], s["restraint"], s["reliability"],
                         s["multi_tool"], s["agent_score"], avg_ms))

        rows.sort(key=lambda r: r[8], reverse=True)

        for rank, (name, backend, mode, origin, action, restraint, reliability,
                   multi_tool, agent_score, avg_ms) in enumerate(rows, 1):
            mt_str = f"{multi_tool:.3f}" if multi_tool is not None else "N/A*"
            print(
                f"{rank:<4} {name:<20} {backend:<12} {mode:<14} {origin:<9} "
                f"{action:>7.3f} {restraint:>10.3f} {reliability:>12.3f} {mt_str:>11} "
                f"{agent_score:>12.3f} {avg_ms:>7}"
            )
    else:
        shdr = (f"{'#':<4} {'Model':<20} {'Origin':<9} {'Tool calls':<12} {'Valid args':<12} "
                f"{'Avg ms':>8} {'Restraint':>10} {'Agent Score':>12}")
        print(shdr)
        print("-" * len(shdr))

        rows = []
        for m in edge_models:
            name = m["name"]
            rs = results[name]
            n_called = sum(1 for r in rs if r["tool_called"])
            n_valid = sum(1 for r in rs if r["valid_args"])
            avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
            restraint_pass = sum(1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"])
            restraint_total = len(RESTRAINT_INDICES)
            score = compute_agent_score(rs)
            origin = _ORIGIN_MAP.get(name, "??")
            rows.append((name, origin, n_called, len(rs), n_valid, n_called, avg_ms,
                          restraint_pass, restraint_total, score))

        rows.sort(key=lambda r: r[9], reverse=True)

        for rank, (name, origin, n_called, total, n_valid, n_called_denom,
                   avg_ms, rpass, rtotal, score) in enumerate(rows, 1):
            print(
                f"{rank:<4} {name:<20} {origin:<9} {n_called:>3}/{total:<8} "
                f"{n_valid:>3}/{n_called_denom if n_called_denom else 0:<8} "
                f"{avg_ms:>7} {rpass:>3}/{rtotal}      {score:>8.3f}"
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

    # all_runs[model_name][run_idx] = [result_per_prompt]
    all_runs = {m["name"]: [] for m in ALL_MODELS}
    # Store raw BitNet responses for logging
    bitnet_raw = {}  # (model_name, run_idx, prompt_idx) -> raw_content

    try:
        for run in range(num_runs):
            print(f"{'='*60}")
            print(f"  RUN {run + 1}/{num_runs}")
            print(f"{'='*60}")
            for model_info in ALL_MODELS:
                name = model_info["name"]
                # Start/switch BitNet server if needed
                if model_info["backend"] == "bitnet":
                    model_path = model_info["model_path"]
                    print(f"  [Starting BitNet server for {name}...]")
                    start_bitnet_server(model_path)
                    print(f"  [BitNet server ready for {name}]")

                print(f"--- {name} ---")
                run_results = []
                for i, prompt in enumerate(TEST_PROMPTS):
                    label = f"  P{i+1}"
                    print(f"{label}: {prompt[:60]}...", end=" ", flush=True)
                    r = run_one(model_info, prompt)
                    tag = r["tool_name"] or ("(no tool)" if not r["error"] else "ERROR")
                    print(f"=> {tag}  [{r['latency_ms']}ms]")
                    run_results.append(r)
                    # Capture raw output for non-native-API backends
                    if model_info["backend"] in ("bitnet", "ollama_raw"):
                        bitnet_raw[(name, run, i)] = r.get("raw_content")
                all_runs[name].append(run_results)
                print()
    finally:
        print("Stopping BitNet server...")
        stop_bitnet_server()
        print("BitNet server stopped.\n")

    # Build averaged results
    model_info_map = {m["name"]: m for m in ALL_MODELS}
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
            # Propagate all_tool_calls: union valid tools appearing in >50% of runs
            all_tc_union = []
            if called:
                tool_counts = {}
                for e in entries:
                    for tc in e.get("all_tool_calls", []):
                        if tc.get("valid") and tc.get("name"):
                            tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1
                for tc_name, count in tool_counts.items():
                    if count > num_runs / 2:
                        all_tc_union.append({"name": tc_name, "valid": True})
            avg_results[name].append({
                "tool_called": called,
                "tool_name": tool_name if called else None,
                "valid_args": valid,
                "latency_ms": avg_lat,
                "error": None,
                "raw_content": None,
                "all_tool_calls": all_tc_union,
            })

    # Compute extended scores
    scores = {}
    for name in model_names:
        mi = model_info_map[name]
        backend_name, mode = get_backend_display(mi)
        scores[name] = {
            "action": compute_action_score(avg_results[name]),
            "restraint": compute_restraint_score(avg_results[name]),
            "reliability": compute_reliability(all_runs[name], num_runs),
            "multi_tool": compute_multi_tool_accuracy(avg_results[name], mi),
            "agent_score": compute_agent_score(avg_results[name]),
            "backend": backend_name,
            "mode": mode,
        }

    # Print per-run detail
    for run in range(num_runs):
        print(f"\n{'='*120}")
        print(f"RUN {run + 1} DETAILS")
        print("=" * 120)
        single = {m["name"]: all_runs[m["name"]][run] for m in ALL_MODELS}
        fmt_table(single, ALL_MODELS)

    # Print per-prompt latency breakdown
    print("\n" + "=" * 120)
    print(f"PER-PROMPT LATENCY ACROSS {num_runs} RUNS (ms)")
    print("=" * 120)
    plabels = [f"P{i+1}" for i in range(len(TEST_PROMPTS))]
    hdr = f"{'Model':<20} {'Prompt':<6} " + "  ".join(f"{'R'+str(r+1):>6}" for r in range(num_runs)) + f"  {'Avg':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name in model_names:
        for pi in range(len(TEST_PROMPTS)):
            vals = [all_runs[name][ri][pi]["latency_ms"] for ri in range(num_runs)]
            avg = round(sum(vals) / num_runs)
            cols = "  ".join(f"{v:>6}" for v in vals)
            print(f"{name:<20} {plabels[pi]:<6} {cols}  {avg:>6}")
        print("-" * len(hdr))

    # Print averaged summary
    print("\n" + "=" * 160)
    print(f"AVERAGED SUMMARY ({num_runs} runs)")
    print("=" * 160)
    fmt_table(avg_results, ALL_MODELS, scores=scores)

    # Print edge agent mini leaderboard
    fmt_edge_leaderboard(avg_results, ALL_MODELS, scores=scores)

    # Print raw BitNet 2B-4T outputs for P1, P6, P8 (indices 0, 5, 7)
    print("\n" + "=" * 120)
    print("RAW BITNET-2B-4T OUTPUT (P1, P6, P8)")
    print("=" * 120)
    for pi, plabel in [(0, "P1"), (5, "P6"), (7, "P8")]:
        print(f"\n--- {plabel}: {TEST_PROMPTS[pi][:70]} ---")
        for run in range(num_runs):
            raw = bitnet_raw.get(("bitnet-2B-4T", run, pi), "(no output)")
            print(f"  Run {run+1}: {raw}")
        print()

    # Also print raw bitnet-3B for comparison
    print("\n" + "=" * 120)
    print("RAW BITNET-3B OUTPUT (P1, P6, P8) [base model, for comparison]")
    print("=" * 120)
    for pi, plabel in [(0, "P1"), (5, "P6"), (7, "P8")]:
        print(f"\n--- {plabel}: {TEST_PROMPTS[pi][:70]} ---")
        for run in range(num_runs):
            raw = bitnet_raw.get(("bitnet-3B", run, pi), "(no output)")
            print(f"  Run {run+1}: {raw}")
        print()


def _self_test():
    """Validate parser and scoring functions against known data."""
    # Test _parse_all_tool_calls_from_text with known BitNet P8 output
    p8_output = (
        '<tool_call>{"name": "search_files", "arguments": {"pattern": "*.py"}}\n'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
    )
    parsed = _parse_all_tool_calls_from_text(p8_output)
    assert len(parsed) == 2, f"Expected 2 tool calls, got {len(parsed)}"
    assert parsed[0]["name"] == "search_files"
    assert parsed[1]["name"] == "get_weather"
    assert all(tc["valid"] for tc in parsed)

    # Test with closing tags
    p8_with_tags = (
        '<tool_call>{"name": "search_files", "arguments": {"pattern": "*.py"}}</tool_call>\n'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )
    parsed2 = _parse_all_tool_calls_from_text(p8_with_tags)
    assert len(parsed2) == 2, f"Expected 2 tool calls with tags, got {len(parsed2)}"

    # Test with one invalid block
    mixed = (
        '<tool_call>{"name": "search_files", "arguments": {"pattern": "*.py"}}\n'
        '<tool_call>invalid json here\n'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
    )
    parsed3 = _parse_all_tool_calls_from_text(mixed)
    assert len(parsed3) == 2, f"Expected 2 valid calls from mixed input, got {len(parsed3)}"

    # Test backward compat: compute_agent_score still produces correct values
    # Simulate a model with 6/7 action correct, 2/2 restraint → 0.929
    mock_results = [
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},   # P1
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},  # P2
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P3
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},   # P4
        {"tool_called": False, "valid_args": None, "tool_name": None},           # P5 restraint
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},   # P6
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P7
        {"tool_called": False, "valid_args": None, "tool_name": None},           # P8 (missed)
        {"tool_called": False, "valid_args": None, "tool_name": None},           # P9 restraint
    ]
    assert compute_agent_score(mock_results) == 0.929, f"Expected 0.929, got {compute_agent_score(mock_results)}"
    assert compute_action_score(mock_results) == 0.857, f"Expected 0.857, got {compute_action_score(mock_results)}"
    assert compute_restraint_score(mock_results) == 1.0, f"Expected 1.0, got {compute_restraint_score(mock_results)}"

    # Test 7/7 action, 0/2 restraint → 0.500 (llama3.2 pattern)
    llama_results = [
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},   # P1
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},  # P2
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P3
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},   # P4
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},  # P5 (should restrain)
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},   # P6
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P7
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},  # P8
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},  # P9 (should restrain)
    ]
    assert compute_agent_score(llama_results) == 0.5, f"Expected 0.5, got {compute_agent_score(llama_results)}"
    assert compute_action_score(llama_results) == 1.0
    assert compute_restraint_score(llama_results) == 0.0

    print("All self-tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
