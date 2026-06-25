from claude_tools import CLAUDE_TOOLS, get_tool_schema


EXPECTED_TOOLS = {
    "get_player_profile",
    "check_inventory",
    "recommend_next_action",
    "generate_quest",
    "validate_quest_completion",
}


def test_claude_tools_include_expected_names():
    names = {tool["name"] for tool in CLAUDE_TOOLS}

    assert names == EXPECTED_TOOLS


def test_each_tool_has_anthropic_required_fields():
    for tool in CLAUDE_TOOLS:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["description"]
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]


def test_get_tool_schema_returns_matching_schema():
    schema = get_tool_schema("check_inventory")

    assert schema["name"] == "check_inventory"
    assert "item_name" in schema["input_schema"]["properties"]


def test_get_tool_schema_returns_none_for_unknown_tool():
    assert get_tool_schema("unknown_tool") is None
