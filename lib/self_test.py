"""Validate parser and scoring functions against known data."""

from lib.bitnet_backend import (
    _parse_all_tool_calls_from_text,
    _parse_tool_call_from_text,
)
from lib.report import (
    compute_action_score,
    compute_restraint_score,
    compute_wrong_tool,
    compute_agent_score,
)


def run():
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

    # Test bare JSON fallback (jan-v3 style: no opening <tool_call> tag)
    bare_json = '{"name": "get_weather", "arguments": {"city": "Antwerp"}}\n</tool_call>'
    parsed_bare = _parse_tool_call_from_text(bare_json)
    assert parsed_bare is not None, "Bare JSON should be parsed"
    assert parsed_bare["name"] == "get_weather"
    assert parsed_bare["arguments"]["city"] == "Antwerp"
    assert parsed_bare["valid"] is True
    bare_all = _parse_all_tool_calls_from_text(bare_json)
    assert len(bare_all) == 1, f"Expected 1 bare JSON call, got {len(bare_all)}"

    # Bare JSON without </tool_call> tag
    bare_no_close = '{"name": "search_files", "arguments": {"pattern": "*.py"}}'
    parsed_bare2 = _parse_tool_call_from_text(bare_no_close)
    assert parsed_bare2 is not None and parsed_bare2["name"] == "search_files"

    # Test bracket notation (lfm2.5 style)
    bracket_single = '[get_weather(city="Antwerp")]I am retrieving the weather.'
    parsed_bracket = _parse_tool_call_from_text(bracket_single)
    assert parsed_bracket is not None, "Bracket notation should be parsed"
    assert parsed_bracket["name"] == "get_weather"
    assert parsed_bracket["arguments"]["city"] == "Antwerp"

    # Multi-tool bracket notation
    bracket_multi = '[search_files(pattern="*.py"), get_weather(city="Paris")]I am searching.'
    bracket_all = _parse_all_tool_calls_from_text(bracket_multi)
    assert len(bracket_all) == 2, f"Expected 2 bracket calls, got {len(bracket_all)}"
    assert bracket_all[0]["name"] == "search_files"
    assert bracket_all[0]["arguments"]["pattern"] == "*.py"
    assert bracket_all[1]["name"] == "get_weather"
    assert bracket_all[1]["arguments"]["city"] == "Paris"

    # Bracket with array argument
    bracket_arr = '[schedule_meeting(title="Sprint Review", time="2pm", attendees=["alice@co.com", "bob@co.com"])]Done.'
    bracket_arr_parsed = _parse_all_tool_calls_from_text(bracket_arr)
    assert len(bracket_arr_parsed) == 1
    assert bracket_arr_parsed[0]["name"] == "schedule_meeting"
    assert bracket_arr_parsed[0]["arguments"]["attendees"] == ["alice@co.com", "bob@co.com"]

    # <tool_call> tag takes priority over bare JSON fallback
    with_tag = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    parsed_tag = _parse_tool_call_from_text(with_tag)
    assert parsed_tag["name"] == "get_weather" and parsed_tag["arguments"]["city"] == "Paris"

    # Plain text should still return None / []
    plain = "The weather in Antwerp is sunny today."
    assert _parse_tool_call_from_text(plain) is None
    assert _parse_all_tool_calls_from_text(plain) == []

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
    run()
