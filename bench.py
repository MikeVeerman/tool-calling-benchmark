#!/usr/bin/env python3
"""Local LLM tool-calling benchmark using Ollama + BitNet."""

import argparse
import json
import os
import sys
import time

import ollama

from lib.bench_config import TEST_PROMPTS, ALL_MODELS, TOOLS
from lib.bitnet_backend import (
    BITNET_SYSTEM_PROMPT,
    start_bitnet_server,
    stop_bitnet_server,
    run_one_bitnet,
    _parse_tool_call_from_text,
    _parse_all_tool_calls_from_text,
)
from lib.report import generate_summary
from lib.run_helpers import (
    compute_bench_version,
    model_name_to_filename,
    find_model,
    save_model_results,
    load_model_results,
    aggregate_runs,
)
from lib.self_test import run as run_self_test

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
        run_self_test()
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


if __name__ == "__main__":
    main()
