#!/usr/bin/env python3
"""Local LLM tool-calling benchmark using Ollama."""

import json
import time
import ollama

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
# Test prompts – from obvious to ambiguous to trick
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # 1 – obvious single-tool
    "What's the weather in Antwerp?",
    # 2 – obvious different tool
    "Find all Python files in the project.",
    # 3 – requires multiple args
    "Schedule a meeting called 'Sprint Review' for 2025-02-10T14:00:00 with alice@co.com and bob@co.com.",
    # 4 – ambiguous, could use tool or not
    "I'm heading to Brussels tomorrow, anything I should know?",
    # 5 – trick / meta question, should NOT call a tool
    "What tools do you have access to?",
]

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

MODELS = [
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "qwen2.5:0.5b",
    "llama3.2:3b",
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_one(model: str, prompt: str) -> dict:
    """Run a single prompt against a model and return result info."""
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

    # Check if args are valid JSON-like (the SDK already parses them, but
    # let's verify they round-trip cleanly)
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


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def fmt_table(results: dict):
    """Print an ASCII summary table."""
    # results: {model: [{...}, ...]}

    # First: per-prompt detail table
    prompt_labels = [f"P{i+1}" for i in range(len(TEST_PROMPTS))]

    print("\n" + "=" * 90)
    print("TEST PROMPTS")
    print("=" * 90)
    for i, p in enumerate(TEST_PROMPTS):
        print(f"  P{i+1}: {p}")

    print("\n" + "=" * 90)
    print("DETAILED RESULTS")
    print("=" * 90)

    hdr = f"{'Model':<18} {'Prompt':<6} {'Called?':<8} {'Tool':<20} {'Args OK':<8} {'ms':>6}"
    print(hdr)
    print("-" * len(hdr))

    for model in MODELS:
        for i, r in enumerate(results[model]):
            called = "YES" if r["tool_called"] else "no"
            tool = r["tool_name"] or "-"
            args_ok = "OK" if r["valid_args"] else ("FAIL" if r["valid_args"] is False else "-")
            ms = str(r["latency_ms"])
            err = f"  ERR: {r['error']}" if r["error"] else ""
            print(f"{model:<18} {prompt_labels[i]:<6} {called:<8} {tool:<20} {args_ok:<8} {ms:>6}{err}")
        print("-" * len(hdr))

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    shdr = f"{'Model':<18} {'Tool calls':<12} {'Valid args':<12} {'Avg ms':>8} {'Trick pass':>11}"
    print(shdr)
    print("-" * len(shdr))

    for model in MODELS:
        rs = results[model]
        n_called = sum(1 for r in rs if r["tool_called"])
        n_valid = sum(1 for r in rs if r["valid_args"])
        avg_ms = round(sum(r["latency_ms"] for r in rs) / len(rs))
        # Trick question (P5) passes if the model did NOT call a tool
        trick = "PASS" if not rs[4]["tool_called"] else "FAIL"
        print(f"{model:<18} {n_called:>3}/{len(rs):<8} {n_valid:>3}/{n_called if n_called else 0:<8} {avg_ms:>7} {trick:>11}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    num_runs = 3
    print(f"Benchmarking {len(MODELS)} models x {len(TEST_PROMPTS)} prompts x {num_runs} runs")
    print(f"Tools available: {', '.join(t['function']['name'] for t in TOOLS)}")
    print()

    # all_runs[model][run_idx] = [result_per_prompt]
    all_runs = {m: [] for m in MODELS}

    for run in range(num_runs):
        print(f"{'='*60}")
        print(f"  RUN {run + 1}/{num_runs}")
        print(f"{'='*60}")
        for model in MODELS:
            print(f"--- {model} ---")
            run_results = []
            for i, prompt in enumerate(TEST_PROMPTS):
                label = f"  P{i+1}"
                print(f"{label}: {prompt[:60]}...", end=" ", flush=True)
                r = run_one(model, prompt)
                tag = r["tool_name"] or ("(no tool)" if not r["error"] else "ERROR")
                print(f"=> {tag}  [{r['latency_ms']}ms]")
                run_results.append(r)
            all_runs[model].append(run_results)
            print()

    # Build averaged results
    # For each model x prompt: average latency, majority-vote tool_called / tool_name / valid_args
    avg_results = {}
    for model in MODELS:
        avg_results[model] = []
        for pi in range(len(TEST_PROMPTS)):
            entries = [all_runs[model][ri][pi] for ri in range(num_runs)]
            avg_lat = round(sum(e["latency_ms"] for e in entries) / num_runs)
            # Majority vote for tool_called
            n_called = sum(1 for e in entries if e["tool_called"])
            called = n_called > num_runs / 2
            # Most common tool name among calls
            tool_names = [e["tool_name"] for e in entries if e["tool_name"]]
            tool_name = max(set(tool_names), key=tool_names.count) if tool_names else None
            # Valid args: count among calls that happened
            n_valid = sum(1 for e in entries if e["valid_args"])
            valid = n_valid > 0 if called else None
            avg_results[model].append({
                "tool_called": called,
                "tool_name": tool_name if called else None,
                "valid_args": valid,
                "latency_ms": avg_lat,
                "error": None,
                "raw_content": None,
            })

    # Print per-run detail
    for run in range(num_runs):
        print(f"\n{'='*90}")
        print(f"RUN {run + 1} DETAILS")
        print("=" * 90)
        single = {m: all_runs[m][run] for m in MODELS}
        fmt_table(single)

    # Print per-prompt latency breakdown
    print("\n" + "=" * 90)
    print(f"PER-PROMPT LATENCY ACROSS {num_runs} RUNS (ms)")
    print("=" * 90)
    plabels = [f"P{i+1}" for i in range(len(TEST_PROMPTS))]
    hdr = f"{'Model':<18} {'Prompt':<6} " + "  ".join(f"{'R'+str(r+1):>6}" for r in range(num_runs)) + f"  {'Avg':>6}"
    print(hdr)
    print("-" * len(hdr))
    for model in MODELS:
        for pi in range(len(TEST_PROMPTS)):
            vals = [all_runs[model][ri][pi]["latency_ms"] for ri in range(num_runs)]
            avg = round(sum(vals) / num_runs)
            cols = "  ".join(f"{v:>6}" for v in vals)
            print(f"{model:<18} {plabels[pi]:<6} {cols}  {avg:>6}")
        print("-" * len(hdr))

    # Print averaged summary
    print("\n" + "=" * 90)
    print(f"AVERAGED SUMMARY ({num_runs} runs)")
    print("=" * 90)
    fmt_table(avg_results)


if __name__ == "__main__":
    main()
