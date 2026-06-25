from types import SimpleNamespace

from npc_dialogue import execute_tool_uses, should_use_claude_tools


def test_should_use_claude_tools_for_tool_worthy_intents():
    assert should_use_claude_tools("inventory_query") is True
    assert should_use_claude_tools("persona_chat") is False
    assert should_use_claude_tools("lore_question") is False
    assert should_use_claude_tools("quest_request") is False
    assert should_use_claude_tools("gm_feedback") is False


def test_execute_tool_uses_returns_tool_results_and_names():
    tool_uses = [
        SimpleNamespace(
            id="toolu_1",
            name="check_inventory",
            input={"player_id": "player_001", "item_name": "月盐药剂"},
        ),
        SimpleNamespace(
            id="toolu_2",
            name="recommend_next_action",
            input={"player_id": "player_001"},
        ),
    ]

    result = execute_tool_uses(tool_uses)

    assert result["tool_names"] == ["check_inventory", "recommend_next_action"]
    assert len(result["tool_results"]) == 2
    assert result["tool_results"][0]["type"] == "tool_result"
    assert result["tool_results"][0]["tool_use_id"] == "toolu_1"
