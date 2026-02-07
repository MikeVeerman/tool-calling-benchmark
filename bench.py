#!/usr/bin/env python3
"""Local LLM tool-calling benchmark using Ollama + BitNet."""

import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

import ollama
import requests

from bitnet_backend import (
    BITNET_SYSTEM_PROMPT,
    start_bitnet_server,
    stop_bitnet_server,
    run_one_bitnet,
    _parse_tool_call_from_text,
    _parse_all_tool_calls_from_text,
)

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
    # P10 – implicit reasoning: cycling decision depends on weather, "weather" never mentioned
    "I have a meeting with a client in Bruges next Thursday. Should I take the train or cycle?",
    # P11 – negation: explicitly says "don't check weather"
    "Don't check the weather in Antwerp, just find me the quarterly report.",
    # P12 – redundant tool trap: weather already provided, just schedule
    "The weather in Antwerp is 8°C and rainy. Should I schedule an indoor meeting with Jan?",
]

# Indices of prompts where the correct behavior is to NOT call a tool
# P5 (idx 4): meta question about tools
# P9 (idx 8): asking to write code, not to call a tool
RESTRAINT_INDICES = {4, 8}

# Indices of prompts where calling a valid tool is clearly correct (for Agent Score)
# P1, P2, P3, P4, P6, P7, P8 are clear tool-call prompts; P10, P11, P12 are hard prompts
TOOL_CALL_INDICES = {0, 1, 2, 3, 5, 6, 7, 9, 10, 11}  # 10 prompts

# P10-P12: expected correct tool for each hard prompt
EXPECTED_TOOLS = {
    9: "get_weather",       # P10: cycling depends on weather (implicit reasoning)
    10: "search_files",     # P11: find the report (negation)
    11: "schedule_meeting", # P12: schedule the meeting (context awareness)
}

# P10-P12: tools that are WRONG (worse than not calling at all)
WRONG_TOOL_MAP = {
    9: {"schedule_meeting"},  # P10: meeting already exists
    10: {"get_weather"},      # P11: explicitly told "don't"
    11: {"get_weather"},      # P12: weather already provided
}

HARD_PROMPT_INDICES = {9, 10, 11}  # P10, P11, P12

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
# Incremental run helpers
# ---------------------------------------------------------------------------


def compute_bench_version() -> str:
    """Hash of prompts + scoring rules. Changes when benchmarks need re-running."""
    content = json.dumps({
        "prompts": TEST_PROMPTS,
        "restraint": sorted(RESTRAINT_INDICES),
        "expected": EXPECTED_TOOLS,
        "wrong": WRONG_TOOL_MAP,
    }, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def model_name_to_filename(name: str) -> str:
    """Convert model name to a safe filename, e.g. 'qwen2.5:3b' -> 'qwen2_5_3b.json'."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return safe + ".json"


def find_model(name: str) -> dict:
    """Look up a model by name in ALL_MODELS. Exit with error if not found."""
    for m in ALL_MODELS:
        if m["name"] == name:
            return m
    available = ", ".join(m["name"] for m in ALL_MODELS)
    print(f"Error: model '{name}' not found.\nAvailable models: {available}")
    sys.exit(1)


def save_model_results(run_dir: str, model_info: dict, runs_data: list[list[dict]], num_runs: int):
    """Write per-model JSON results file."""
    filepath = os.path.join(run_dir, model_name_to_filename(model_info["name"]))
    payload = {
        "model_name": model_info["name"],
        "model_info": model_info,
        "bench_version": compute_bench_version(),
        "num_runs": num_runs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": runs_data,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)


def load_model_results(filepath: str) -> dict:
    """Load a per-model JSON results file."""
    with open(filepath) as f:
        return json.load(f)


def aggregate_runs(runs_for_model: list[list[dict]], num_runs: int) -> list[dict]:
    """Majority-vote aggregation across runs for one model. Returns one result per prompt."""
    aggregated = []
    for pi in range(len(TEST_PROMPTS)):
        entries = [runs_for_model[ri][pi] for ri in range(num_runs)]
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
        aggregated.append({
            "tool_called": called,
            "tool_name": tool_name if called else None,
            "valid_args": valid,
            "latency_ms": avg_lat,
            "error": None,
            "raw_content": None,
            "all_tool_calls": all_tc_union,
        })
    return aggregated


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
    """Action Score = correct_tool_calls / 10 (actionable prompts).

    For P10-P12, the specific expected tool must be called.
    """
    rs = results_for_model
    count = 0
    for idx in TOOL_CALL_INDICES:
        if idx in EXPECTED_TOOLS:
            if rs[idx]["valid_args"] and rs[idx]["tool_name"] == EXPECTED_TOOLS[idx]:
                count += 1
        else:
            if rs[idx]["valid_args"]:
                count += 1
    return round(count / len(TOOL_CALL_INDICES), 3)


def compute_restraint_score(results_for_model: list[dict]) -> float:
    """Restraint Score = correct_refusals / 2 (restraint prompts P5, P9)."""
    rs = results_for_model
    restraint_pass = sum(1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"])
    return round(restraint_pass / len(RESTRAINT_INDICES), 3)


def compute_wrong_tool(results_for_model: list[dict]) -> int:
    """Count wrong tool calls across P10-P12."""
    rs = results_for_model
    count = 0
    for idx, wrong_tools in WRONG_TOOL_MAP.items():
        if rs[idx]["tool_called"] and rs[idx]["tool_name"] in wrong_tools:
            count += 1
    return count


def compute_agent_score(results_for_model: list[dict]) -> float:
    """Agent Score = Action * 0.4 + Restraint * 0.3 + Wrong-Tool-Avoidance * 0.3.

    Uses raw (unrounded) values to avoid double-rounding.
    """
    rs = results_for_model
    # Action: correct tool calls / 10 (with expected-tool matching for P10-P12)
    action_count = 0
    for idx in TOOL_CALL_INDICES:
        if idx in EXPECTED_TOOLS:
            if rs[idx]["valid_args"] and rs[idx]["tool_name"] == EXPECTED_TOOLS[idx]:
                action_count += 1
        else:
            if rs[idx]["valid_args"]:
                action_count += 1
    accuracy = action_count / len(TOOL_CALL_INDICES)
    # Restraint: correct refusals / 2
    restraint_pass = sum(1 for idx in RESTRAINT_INDICES if not rs[idx]["tool_called"])
    restraint = restraint_pass / len(RESTRAINT_INDICES)
    # Wrong tool avoidance: (3 - wrong_tool) / 3
    wrong = compute_wrong_tool(rs)
    wrong_avoidance = (3 - wrong) / 3
    return round(accuracy * 0.4 + restraint * 0.3 + wrong_avoidance * 0.3, 3)


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
            elif pi in EXPECTED_TOOLS:
                if r["valid_args"] and r["tool_name"] == EXPECTED_TOOLS[pi]:
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
        tag = ""
        if i in RESTRAINT_INDICES:
            tag = " [RESTRAINT]"
        elif i in HARD_PROMPT_INDICES:
            tag = " [HARD]"
        print(f"  P{i+1}: {p}{tag}")

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
                f"{'Action':>7} {'Restraint':>10} {'Wrong Tool':>11} {'Reliability':>12} {'Multi-Tool':>11} "
                f"{'Agent Score':>12} {'Avg ms':>8}")
        print(shdr)
        print("-" * len(shdr))

        rows = []
        for name in model_names:
            rs = results[name]
            avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
            s = scores[name]
            rows.append((name, s["backend"], s["mode"], _ORIGIN_MAP.get(name, "??"),
                         s["action"], s["restraint"], s["wrong_tool"],
                         s["reliability"], s["multi_tool"], s["agent_score"], avg_ms))

        rows.sort(key=lambda r: r[9], reverse=True)

        for (name, backend, mode, origin, action, restraint, wrong_tool,
             reliability, multi_tool, agent_score, avg_ms) in rows:
            mt_str = f"{multi_tool:.3f}" if multi_tool is not None else "N/A*"
            print(
                f"{name:<20} {backend:<12} {mode:<14} {origin:<9} "
                f"{action:>7.3f} {restraint:>10.3f} {wrong_tool:>11} {reliability:>12.3f} {mt_str:>11} "
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
                f"{'Action':>7} {'Restraint':>10} {'Wrong Tool':>11} {'Reliability':>12} {'Multi-Tool':>11} "
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
                         s["action"], s["restraint"], s["wrong_tool"],
                         s["reliability"], s["multi_tool"], s["agent_score"], avg_ms))

        rows.sort(key=lambda r: r[9], reverse=True)

        for rank, (name, backend, mode, origin, action, restraint, wrong_tool,
                   reliability, multi_tool, agent_score, avg_ms) in enumerate(rows, 1):
            mt_str = f"{multi_tool:.3f}" if multi_tool is not None else "N/A*"
            print(
                f"{rank:<4} {name:<20} {backend:<12} {mode:<14} {origin:<9} "
                f"{action:>7.3f} {restraint:>10.3f} {wrong_tool:>11} {reliability:>12.3f} {mt_str:>11} "
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


def fmt_hard_prompts_table(results: dict, model_list: list[dict]):
    """Print a focused table showing P10/P11/P12 results per model."""
    model_names = [m["name"] for m in model_list]

    print("\n" + "=" * 120)
    print("HARD PROMPTS P10-P12 (which tool did each model call?)")
    print("=" * 120)
    print(f"  P10: {TEST_PROMPTS[9][:80]}")
    print(f"       Expected: get_weather | Wrong: schedule_meeting")
    print(f"  P11: {TEST_PROMPTS[10][:80]}")
    print(f"       Expected: search_files | Wrong: get_weather")
    print(f"  P12: {TEST_PROMPTS[11][:80]}")
    print(f"       Expected: schedule_meeting | Wrong: get_weather")
    print()

    shdr = (f"{'Model':<20} {'P10 Tool':<20} {'P10':<8} "
            f"{'P11 Tool':<20} {'P11':<8} "
            f"{'P12 Tool':<20} {'P12':<8} {'Wrong':>6}")
    print(shdr)
    print("-" * len(shdr))

    for name in model_names:
        rs = results[name]
        cols = []
        wrong_count = 0
        for idx in [9, 10, 11]:
            tool = rs[idx]["tool_name"] or "(none)"
            expected = EXPECTED_TOOLS[idx]
            wrong_tools = WRONG_TOOL_MAP[idx]
            if rs[idx]["tool_called"] and rs[idx]["tool_name"] == expected:
                verdict = "OK"
            elif rs[idx]["tool_called"] and rs[idx]["tool_name"] in wrong_tools:
                verdict = "WRONG"
                wrong_count += 1
            elif rs[idx]["tool_called"]:
                verdict = "wrong?"
                wrong_count += 1
            else:
                verdict = "miss"
            cols.append((tool, verdict))
        print(f"{name:<20} {cols[0][0]:<20} {cols[0][1]:<8} "
              f"{cols[1][0]:<20} {cols[1][1]:<8} "
              f"{cols[2][0]:<20} {cols[2][1]:<8} {wrong_count:>6}")

    print()


# ---------------------------------------------------------------------------
# TeeWriter – duplicates print() output to stdout + a file
# ---------------------------------------------------------------------------


class TeeWriter(io.TextIOBase):
    """Write to both the real stdout and a file handle."""

    def __init__(self, file_handle):
        self._stdout = sys.stdout
        self._file = file_handle

    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)
        return len(s)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ---------------------------------------------------------------------------
# Single-model runner
# ---------------------------------------------------------------------------


def run_single_model(model_info: dict, num_runs: int, run_dir: str):
    """Run one model for N iterations, print progress, save JSON to run_dir."""
    name = model_info["name"]
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running {name} x {len(TEST_PROMPTS)} prompts x {num_runs} runs")
    print(f"Output: {run_dir}")
    print()

    runs_data = []  # runs_data[run_idx] = [result_per_prompt]

    try:
        # Start BitNet server if needed
        if model_info["backend"] == "bitnet":
            model_path = model_info["model_path"]
            print(f"  [Starting BitNet server for {name}...]")
            start_bitnet_server(model_path)
            print(f"  [BitNet server ready for {name}]")

        for run in range(num_runs):
            print(f"{'='*60}")
            print(f"  {name} — RUN {run + 1}/{num_runs}")
            print(f"{'='*60}")
            run_results = []
            for i, prompt in enumerate(TEST_PROMPTS):
                print(f"  P{i+1}: {prompt[:60]}...", end=" ", flush=True)
                r = run_one(model_info, prompt)
                tag = r["tool_name"] or ("(no tool)" if not r["error"] else "ERROR")
                print(f"=> {tag}  [{r['latency_ms']}ms]")
                run_results.append(r)
            runs_data.append(run_results)
            print()
    finally:
        if model_info["backend"] == "bitnet":
            print("Stopping BitNet server...")
            stop_bitnet_server()
            print("BitNet server stopped.\n")

    save_model_results(run_dir, model_info, runs_data, num_runs)
    print(f"Saved {model_name_to_filename(name)}")


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(run_dir: str):
    """Read all per-model JSON from run_dir, compute scores, write summary.txt."""
    json_files = sorted(glob.glob(os.path.join(run_dir, "*.json")))
    if not json_files:
        print(f"No model result files found in {run_dir}/")
        return

    current_version = compute_bench_version()
    stale_models = []

    # Load all model data
    model_data = {}  # name -> loaded dict
    for fp in json_files:
        data = load_model_results(fp)
        model_data[data["model_name"]] = data

    # Build model_list in the order they appear in ALL_MODELS, skipping missing
    all_model_names = [m["name"] for m in ALL_MODELS]
    model_list = []
    for m in ALL_MODELS:
        if m["name"] in model_data:
            model_list.append(m)

    # Also include any models in JSON that aren't in ALL_MODELS (future-proof)
    known_names = {m["name"] for m in ALL_MODELS}
    for name, data in model_data.items():
        if name not in known_names:
            model_list.append(data["model_info"])

    model_names = [m["name"] for m in model_list]

    # Aggregate and score each model
    avg_results = {}
    all_runs = {}
    scores = {}
    for name in model_names:
        data = model_data[name]
        num_runs = data["num_runs"]
        runs = data["runs"]
        all_runs[name] = runs
        avg_results[name] = aggregate_runs(runs, num_runs)

        mi = data["model_info"]
        backend_name, mode = get_backend_display(mi)
        scores[name] = {
            "action": compute_action_score(avg_results[name]),
            "restraint": compute_restraint_score(avg_results[name]),
            "wrong_tool": compute_wrong_tool(avg_results[name]),
            "reliability": compute_reliability(runs, num_runs),
            "multi_tool": compute_multi_tool_accuracy(avg_results[name], mi),
            "agent_score": compute_agent_score(avg_results[name]),
            "backend": backend_name,
            "mode": mode,
        }

        if data.get("bench_version") != current_version:
            stale_models.append(name)

    # Write summary to summary.txt (and stdout)
    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, "w") as sf:
        tee = TeeWriter(sf)
        with contextlib.redirect_stdout(tee):
            # Averaged summary
            print("=" * 160)
            print(f"SUMMARY ({len(model_names)} models)")
            print("=" * 160)

            # If stale models, add asterisks to names for display
            display_results = avg_results
            display_scores = scores
            if stale_models:
                display_results = {}
                display_scores = {}
                for name in model_names:
                    dname = name + "*" if name in stale_models else name
                    display_results[dname] = avg_results[name]
                    display_scores[dname] = scores[name]
                # Also need display model_list with starred names
                display_model_list = []
                for m in model_list:
                    dm = dict(m)
                    if m["name"] in stale_models:
                        dm["name"] = m["name"] + "*"
                    display_model_list.append(dm)
                # Update _ORIGIN_MAP temporarily
                for m in display_model_list:
                    if m["name"] not in _ORIGIN_MAP:
                        _ORIGIN_MAP[m["name"]] = m.get("origin", "??")
            else:
                display_model_list = model_list

            fmt_table(display_results, display_model_list, scores=display_scores)

            # Edge agent mini leaderboard
            fmt_edge_leaderboard(display_results, display_model_list, scores=display_scores)

            # Hard prompts P10-P12 focused table
            fmt_hard_prompts_table(display_results, display_model_list)

            if stale_models:
                print("* Stale results (bench_version mismatch): " + ", ".join(stale_models))
                print(f"  Current bench_version: {current_version}")
                print("  Re-run these models to update.\n")

    print(f"\nSummary written to {summary_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Local LLM tool-calling benchmark",
        usage="%(prog)s [model] [options]",
    )
    parser.add_argument("model", nargs="?", help="Model name to benchmark (e.g. qwen2.5:3b)")
    parser.add_argument("--all", action="store_true", help="Run all stale/missing models")
    parser.add_argument("--force", action="store_true", help="With --all, rerun everything")
    parser.add_argument("--summary", action="store_true", help="Regenerate summary from saved results")
    parser.add_argument("--list", action="store_true", dest="list_models", help="List models + staleness status")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per model (default: 3)")
    parser.add_argument("--run-dir", default=None, help="Run directory (default: runs/default/)")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    # Resolve run directory
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = args.run_dir or os.path.join(bench_dir, "runs", "default")

    if args.self_test:
        _self_test()
        return

    if args.list_models:
        current_version = compute_bench_version()
        print(f"Bench version: {current_version}")
        print(f"Run directory: {run_dir}\n")
        for m in ALL_MODELS:
            fp = os.path.join(run_dir, model_name_to_filename(m["name"]))
            if not os.path.exists(fp):
                status = "[missing]"
            else:
                data = load_model_results(fp)
                if data.get("bench_version") == current_version:
                    status = "[ok]"
                else:
                    status = "[stale]"
            print(f"  {m['name']:<24} {status}")
        return

    if args.summary:
        if not os.path.isdir(run_dir):
            print(f"Run directory not found: {run_dir}")
            sys.exit(1)
        generate_summary(run_dir)
        return

    if args.all:
        current_version = compute_bench_version()
        models_to_run = []
        for m in ALL_MODELS:
            fp = os.path.join(run_dir, model_name_to_filename(m["name"]))
            if args.force or not os.path.exists(fp):
                models_to_run.append(m)
            else:
                data = load_model_results(fp)
                if data.get("bench_version") != current_version:
                    models_to_run.append(m)
        if not models_to_run:
            print("All models are up to date. Use --force to rerun.")
        else:
            print(f"Running {len(models_to_run)} model(s): {', '.join(m['name'] for m in models_to_run)}\n")
            for m in models_to_run:
                run_single_model(m, args.num_runs, run_dir)
        generate_summary(run_dir)
        return

    if args.model:
        model_info = find_model(args.model)
        run_single_model(model_info, args.num_runs, run_dir)
        generate_summary(run_dir)
        return

    parser.print_help()


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

    # Test scoring with new 12-prompt formula
    # Good model: 9/10 action (misses P8), 2/2 restraint, 0/3 wrong tool
    # agent_score = (9/10)*0.4 + (2/2)*0.3 + ((3-0)/3)*0.3 = 0.36+0.3+0.3 = 0.96
    mock_results = [
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P1
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},      # P2
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P3
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P4
        {"tool_called": False, "valid_args": None, "tool_name": None},               # P5 restraint
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P6
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P7
        {"tool_called": False, "valid_args": None, "tool_name": None},               # P8 (missed)
        {"tool_called": False, "valid_args": None, "tool_name": None},               # P9 restraint
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P10 correct
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},      # P11 correct
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P12 correct
    ]
    assert compute_agent_score(mock_results) == 0.96, f"Expected 0.96, got {compute_agent_score(mock_results)}"
    assert compute_action_score(mock_results) == 0.9, f"Expected 0.9, got {compute_action_score(mock_results)}"
    assert compute_restraint_score(mock_results) == 1.0, f"Expected 1.0, got {compute_restraint_score(mock_results)}"
    assert compute_wrong_tool(mock_results) == 0, f"Expected 0 wrong, got {compute_wrong_tool(mock_results)}"

    # Trigger-happy model: 7/10 action, 0/2 restraint, 3/3 wrong tool
    # agent_score = (7/10)*0.4 + (0/2)*0.3 + ((3-3)/3)*0.3 = 0.28+0+0 = 0.28
    llama_results = [
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P1
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},      # P2
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P3
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P4
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},      # P5 (should restrain)
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P6
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P7
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},      # P8
        {"tool_called": True, "valid_args": True, "tool_name": "search_files"},      # P9 (should restrain)
        {"tool_called": True, "valid_args": True, "tool_name": "schedule_meeting"},  # P10 WRONG
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P11 WRONG
        {"tool_called": True, "valid_args": True, "tool_name": "get_weather"},       # P12 WRONG
    ]
    assert compute_agent_score(llama_results) == 0.28, f"Expected 0.28, got {compute_agent_score(llama_results)}"
    assert compute_action_score(llama_results) == 0.7, f"Expected 0.7, got {compute_action_score(llama_results)}"
    assert compute_restraint_score(llama_results) == 0.0
    assert compute_wrong_tool(llama_results) == 3, f"Expected 3 wrong, got {compute_wrong_tool(llama_results)}"

    print("All self-tests passed.")


if __name__ == "__main__":
    main()
