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

# Known tool names and their parameter names (for positional-arg fallback)
KNOWN_TOOLS = {
    "get_weather": ["city"],
    "search_files": ["pattern"],
    "schedule_meeting": ["title", "time", "attendees"],
}

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


def _parse_positional_value(args_str: str, pos: int) -> tuple:
    """Parse a single positional value starting at pos. Returns (value, end_pos) or (None, pos)."""
    if pos >= len(args_str):
        return None, pos
    ch = args_str[pos]
    if ch in ('"', "'"):
        end = pos + 1
        while end < len(args_str) and args_str[end] != ch:
            end += 1
        return args_str[pos + 1:end], end + 1 if end < len(args_str) else end
    if ch == "[":
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
            return json.loads(arr_str), end
        except json.JSONDecodeError:
            return arr_str, end
    # Bare value — read until comma or end
    end = pos
    while end < len(args_str) and args_str[end] not in ",)":
        end += 1
    val = args_str[pos:end].strip()
    if not val:
        return None, pos
    try:
        return int(val), end
    except ValueError:
        try:
            return float(val), end
        except ValueError:
            return val, end


def _parse_bracket_args(args_str: str, param_names: list[str] | None = None) -> dict:
    """Parse keyword arguments from bracket notation: key="value", key=val, key: val, key=[list].

    Also handles positional arguments (bare values without key) when param_names
    is provided — assigns them to parameter names in order.
    """
    args = {}
    pos = 0
    positional_idx = 0
    while pos < len(args_str):
        # Skip whitespace and commas between args
        while pos < len(args_str) and args_str[pos] in " ,\t\n":
            pos += 1
        if pos >= len(args_str):
            break
        # Match key= or key: (colon-separated for gemma3-style)
        km = re.match(r"(\w+)\s*[=:]\s*", args_str[pos:])
        if not km:
            # No key — try positional argument
            val, end = _parse_positional_value(args_str, pos)
            if val is not None and param_names and positional_idx < len(param_names):
                args[param_names[positional_idx]] = val
                positional_idx += 1
                pos = end
                continue
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


def _parse_funcall(text: str) -> dict | None:
    """Parse a single function-call like 'get_weather(city: Antwerp)' or 'get_weather(Antwerp)'.

    Returns {"name": ..., "arguments": {...}, "valid": True} or None.
    """
    m = re.match(r"(\w+)\(", text.strip())
    if not m:
        return None
    fname = m.group(1)
    paren_start = m.end()
    # Find matching closing paren
    pdepth = 1
    in_s = False
    s_ch = None
    paren_end = -1
    for j in range(paren_start, len(text)):
        c = text[j]
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
        return None
    args_str = text[paren_start:paren_end]
    param_names = KNOWN_TOOLS.get(fname)
    parsed_args = _parse_bracket_args(args_str, param_names=param_names)
    return {"name": fname, "arguments": parsed_args, "valid": True}


_TYPE_KEYWORDS = {"string", "str", "int", "float", "bool", "list", "array", "string[]", "number", "object"}


def _is_type_signature(args: dict) -> bool:
    """Check if parsed args look like a type signature (e.g. city: string) rather than real values."""
    if not args:
        return False
    for k, v in args.items():
        if not isinstance(v, str):
            return False
        # Value is a type keyword
        if v.lower() in _TYPE_KEYWORDS:
            continue
        # Value equals the parameter name (e.g. city: city)
        if v.lower() == k.lower():
            continue
        # Value looks like a placeholder (e.g. city_name, your_city)
        if k.lower() in v.lower().replace("_", " ").replace("-", " "):
            continue
        return False
    return True


def _parse_bare_funcall_tool_calls(content: str) -> list[dict]:
    """Fallback: parse bare function calls like 'get_weather(city: "Antwerp")' with no tags.

    Only matches known tool names to avoid false positives on Python code.
    Skips: matches preceded by 'def '/'.'/'= ', empty args, and type signatures.
    """
    results = []
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in KNOWN_TOOLS) + r")\(")
    pos = 0
    while pos < len(content):
        m = pattern.search(content, pos)
        if not m:
            break
        # Skip if preceded by 'def ', '.', or '= ' (Python code patterns)
        prefix_start = max(0, m.start() - 4)
        prefix = content[prefix_start:m.start()]
        if prefix.endswith("def ") or prefix.endswith(".") or prefix.endswith("= "):
            pos = m.end()
            continue
        parsed = _parse_funcall(content[m.start():])
        if parsed and not _is_type_signature(parsed["arguments"]) and parsed["arguments"]:
            results.append(parsed)
            # Advance past this call
            pos = m.start() + 1
            # Skip ahead to after the closing paren
            trial = content[m.start():]
            pm = re.match(r"\w+\(", trial)
            if pm:
                pdepth = 1
                for j in range(pm.end(), len(trial)):
                    if trial[j] == "(":
                        pdepth += 1
                    elif trial[j] == ")":
                        pdepth -= 1
                        if pdepth == 0:
                            pos = m.start() + j + 1
                            break
        else:
            pos = m.start() + 1
    return results


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fence markers (```json, etc.) leaving inner content for JSON parsing."""
    return re.sub(r"```\w*\s*\n?", "", content)


def _remove_code_blocks(content: str) -> str:
    """Remove entire fenced code blocks from content (for bare funcall to avoid Python code)."""
    return re.sub(r"```\w*\n.*?```", "", content, flags=re.DOTALL)


def _parse_tag_funcall(content: str) -> dict | None:
    """Parse function-call syntax inside <tool_call> tags: <tool_call>fn(args)</tool_call>."""
    idx = content.find("<tool_call>")
    if idx == -1:
        return None
    rest = content[idx + len("<tool_call>"):].lstrip()
    # Strip optional closing tag for the boundary
    close = rest.find("</tool_call>")
    if close != -1:
        rest = rest[:close].rstrip()
    return _parse_funcall(rest)


def _parse_all_tag_funcalls(content: str) -> list[dict]:
    """Parse all function-call syntax inside <tool_call> tags."""
    results = []
    search_start = 0
    while True:
        idx = content.find("<tool_call>", search_start)
        if idx == -1:
            break
        rest = content[idx + len("<tool_call>"):].lstrip()
        # Determine boundary: next <tool_call> or </tool_call>
        close = rest.find("</tool_call>")
        next_open = rest.find("<tool_call>")
        if close != -1 and (next_open == -1 or close < next_open):
            segment = rest[:close].rstrip()
            search_start = idx + len("<tool_call>") + close + len("</tool_call>")
        elif next_open != -1:
            segment = rest[:next_open].rstrip()
            search_start = idx + len("<tool_call>") + next_open
        else:
            segment = rest.rstrip()
            search_start = len(content)
        parsed = _parse_funcall(segment)
        if parsed:
            results.append(parsed)
    return results


def _parse_tool_call_from_text(content: str) -> dict | None:
    """Parse tool call from raw text. Primary: <tool_call> tags. Fallbacks: bare JSON, bracket notation, bare funcall."""
    idx = content.find("<tool_call>")
    if idx != -1:
        rest = content[idx + len("<tool_call>"):].lstrip()
        if rest.startswith("{"):
            # Standard JSON inside <tool_call> tags
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
        # Not JSON — try function-call syntax inside tags
        result = _parse_tag_funcall(content)
        if result:
            return result
        return None
    # No <tool_call> tag — try fallbacks
    stripped = _strip_code_fences(content)
    result = _parse_bare_json_tool_call(stripped)
    if result:
        return result
    bracket_calls = _parse_bracket_tool_calls(content)
    if bracket_calls:
        return bracket_calls[0]
    # Bare funcall: strip code blocks first to avoid matching Python code examples
    no_code = _remove_code_blocks(content)
    bare_calls = _parse_bare_funcall_tool_calls(no_code)
    if bare_calls:
        return bare_calls[0]
    return None


def _parse_all_tool_calls_from_text(content: str) -> list[dict]:
    """Parse ALL tool call blocks from raw text. Returns list of parsed dicts.

    Handles sequential <tool_call> blocks (JSON or function-call syntax),
    bare JSON, bracket notation, and bare function calls.
    """
    results = []
    has_tag = "<tool_call>" in content
    if has_tag:
        # Try JSON inside tags first
        search_start = 0
        while True:
            idx = content.find("<tool_call>", search_start)
            if idx == -1:
                break
            rest = content[idx + len("<tool_call>"):].lstrip()
            if not rest.startswith("{"):
                search_start = idx + len("<tool_call>")
                continue
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
                json.dumps(args)
                results.append({"name": fname, "arguments": args, "valid": True})
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            search_start = idx + len("<tool_call>") + end
        if results:
            return results
        # No JSON found in tags — try function-call syntax in tags
        funcall_results = _parse_all_tag_funcalls(content)
        if funcall_results:
            return funcall_results
        return results  # empty — tags present but unparseable
    # No <tool_call> tags — try fallbacks in order
    stripped = _strip_code_fences(content)
    bare = _parse_bare_json_all_tool_calls(stripped)
    if bare:
        return bare
    bracket = _parse_bracket_tool_calls(content)
    if bracket:
        return bracket
    no_code = _remove_code_blocks(content)
    return _parse_bare_funcall_tool_calls(no_code)


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
