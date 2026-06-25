from tool_executor import execute_tool_call, tool_result_block


def test_execute_tool_call_runs_check_inventory():
    result = execute_tool_call(
        {
            "name": "check_inventory",
            "input": {"player_id": "player_001", "item_name": "月盐药剂"},
        }
    )

    assert result["ok"] is True
    assert result["tool_name"] == "check_inventory"
    assert result["result"]["has_item"] is True


def test_execute_tool_call_runs_recommend_next_action():
    result = execute_tool_call(
        {
            "name": "recommend_next_action",
            "input": {"player_id": "player_001"},
        }
    )

    assert result["ok"] is True
    assert "recommendation" in result["result"]


def test_execute_tool_call_reports_unknown_tool():
    result = execute_tool_call({"name": "unknown_tool", "input": {}})

    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_execute_tool_call_reports_missing_argument():
    result = execute_tool_call({"name": "check_inventory", "input": {"player_id": "player_001"}})

    assert result["ok"] is False
    assert "missing required argument" in result["error"]


def test_execute_tool_call_ignores_extra_arguments():
    result = execute_tool_call(
        {
            "name": "check_inventory",
            "input": {
                "player_id": "player_001",
                "item_name": "月盐药剂",
                "unused": "ignored",
            },
        }
    )

    assert result["ok"] is True
    assert result["result"]["has_item"] is True


def test_execute_tool_call_rejects_non_object_input():
    result = execute_tool_call({"name": "check_inventory", "input": "月盐药剂"})

    assert result["ok"] is False
    assert "tool input must be an object" in result["error"]


def test_tool_result_block_serializes_result():
    execution = execute_tool_call(
        {
            "name": "check_inventory",
            "input": {"player_id": "player_001", "item_name": "月盐药剂"},
        }
    )

    block = tool_result_block("toolu_123", execution)

    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_123"
    assert "月盐药剂" in block["content"]


def test_tool_result_block_marks_failed_execution_as_error():
    execution = execute_tool_call({"name": "unknown_tool", "input": {}})

    block = tool_result_block("toolu_bad", execution)

    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_bad"
    assert block["is_error"] is True
