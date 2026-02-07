"""Scoring, table formatting, and summary generation."""

import contextlib
import glob
import io
import os
import sys

from bench_config import (
    TEST_PROMPTS,
    RESTRAINT_INDICES,
    TOOL_CALL_INDICES,
    EXPECTED_TOOLS,
    WRONG_TOOL_MAP,
    HARD_PROMPT_INDICES,
    P8_REQUIRED_TOOLS,
    ALL_MODELS,
    EDGE_MODELS,
    get_backend_display,
)
from run_helpers import (
    compute_bench_version,
    load_model_results,
    aggregate_runs,
)


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
