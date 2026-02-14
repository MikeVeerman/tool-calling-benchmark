"""BitNet backend: server lifecycle, runner, and text-based tool-call parsing."""

import json
import re
import subprocess
import time

import requests

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


def _parse_bare_json_tool_call(content: str) -> dict | None:
    """Fallback: parse bare JSON object with "name" and "arguments" keys."""
    idx = 0
    while idx < len(content):
        brace = content.find("{", idx)
        if brace == -1:
            break
        depth = 0
        end = -1
        for i in range(brace, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            break
        try:
            call = json.loads(content[brace:end])
            if isinstance(call, dict) and "name" in call and "arguments" in call:
                args = call["arguments"]
                json.dumps(args)  # validate serialisable
                return {"name": call["name"], "arguments": args, "valid": True}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        idx = brace + 1
    return None


def _parse_bare_json_all_tool_calls(content: str) -> list[dict]:
    """Fallback: parse all bare JSON tool call objects from text."""
    results = []
    idx = 0
    while idx < len(content):
        brace = content.find("{", idx)
        if brace == -1:
            break
        depth = 0
        end = -1
        for i in range(brace, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            break
        try:
            call = json.loads(content[brace:end])
            if isinstance(call, dict) and "name" in call and "arguments" in call:
                args = call["arguments"]
                json.dumps(args)
                results.append({"name": call["name"], "arguments": args, "valid": True})
                idx = end
                continue
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        idx = brace + 1
    return results


def _parse_bracket_args(args_str: str) -> dict:
    """Parse keyword arguments from bracket notation: key="value", key=[list]."""
    args = {}
    pos = 0
    while pos < len(args_str):
        # Skip whitespace and commas between args
        while pos < len(args_str) and args_str[pos] in " ,\t\n":
            pos += 1
        if pos >= len(args_str):
            break
        # Match key=
        km = re.match(r"(\w+)\s*=\s*", args_str[pos:])
        if not km:
            break
        key = km.group(1)
        pos += km.end()
        if pos >= len(args_str):
            break
        ch = args_str[pos]
        if ch in ('"', "'"):
            # Quoted string: find matching close quote
            end = pos + 1
            while end < len(args_str) and args_str[end] != ch:
                end += 1
            args[key] = args_str[pos + 1:end]
            pos = end + 1 if end < len(args_str) else end
        elif ch == "[":
            # Array: find matching ]
            depth = 1
            end = pos + 1
            while end < len(args_str) and depth > 0:
                if args_str[end] == "[":
                    depth += 1
                elif args_str[end] == "]":
                    depth -= 1
                end += 1
            arr_str = args_str[pos:end]
            try:
                args[key] = json.loads(arr_str)
            except json.JSONDecodeError:
                args[key] = arr_str
            pos = end
        else:
            # Bare value (number, etc.)
            end = pos
            while end < len(args_str) and args_str[end] not in ",)":
                end += 1
            val = args_str[pos:end].strip()
            try:
                args[key] = int(val)
            except ValueError:
                try:
                    args[key] = float(val)
                except ValueError:
                    args[key] = val
            pos = end
    return args


def _parse_bracket_tool_calls(content: str) -> list[dict]:
    """Fallback: parse bracket-notation tool calls like [fn(arg="val")].

    Handles formats like:
        [get_weather(city="Antwerp")]
        [search_files(pattern="*.py"), get_weather(city="Paris")]
    """
    m = re.search(r"\[(\w+)\(", content)
    if not m:
        return []
    start = m.start()
    # Find matching closing bracket (track depth, skip strings)
    depth = 0
    end = -1
    in_str = False
    str_ch = None
    for i in range(start, len(content)):
        c = content[i]
        if in_str:
            if c == str_ch:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                str_ch = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end == -1:
        return []
    inner = content[start + 1:end]

    # Find each function_name(args) call within the brackets
    results = []
    call_re = re.compile(r"(\w+)\(")
    pos = 0
    while pos < len(inner):
        cm = call_re.search(inner, pos)
        if not cm:
            break
        fname = cm.group(1)
        paren_start = cm.end()
        # Find matching closing paren (skip strings)
        pdepth = 1
        in_s = False
        s_ch = None
        paren_end = -1
        for j in range(paren_start, len(inner)):
            c = inner[j]
            if in_s:
                if c == s_ch:
                    in_s = False
            else:
                if c in ('"', "'"):
                    in_s = True
                    s_ch = c
                elif c == "(":
                    pdepth += 1
                elif c == ")":
                    pdepth -= 1
                    if pdepth == 0:
                        paren_end = j
                        break
        if paren_end == -1:
            pos = paren_start
            continue
        args_str = inner[paren_start:paren_end]
        parsed_args = _parse_bracket_args(args_str)
        results.append({"name": fname, "arguments": parsed_args, "valid": True})
        pos = paren_end + 1
    return results


def _parse_tool_call_from_text(content: str) -> dict | None:
    """Parse tool call from raw text. Primary: <tool_call> tags. Fallbacks: bare JSON, bracket notation."""
    idx = content.find("<tool_call>")
    if idx != -1:
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
    # No <tool_call> tag — try fallbacks
    result = _parse_bare_json_tool_call(content)
    if result:
        return result
    bracket_calls = _parse_bracket_tool_calls(content)
    if bracket_calls:
        return bracket_calls[0]
    return None


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
    if results or "<tool_call>" in content:
        return results
    # No <tool_call> tags — try fallbacks
    bare = _parse_bare_json_all_tool_calls(content)
    if bare:
        return bare
    return _parse_bracket_tool_calls(content)


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
