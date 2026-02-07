"""Incremental run helpers: versioning, file I/O, aggregation."""

import hashlib
import json
import os
import re
import sys
from datetime import datetime

from lib.bench_config import (
    TEST_PROMPTS,
    RESTRAINT_INDICES,
    EXPECTED_TOOLS,
    WRONG_TOOL_MAP,
    ALL_MODELS,
)


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
    num_prompts = len(runs_for_model[0]) if runs_for_model else 0
    for pi in range(num_prompts):
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
