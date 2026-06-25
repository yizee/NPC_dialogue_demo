from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from game_tools import (
    check_inventory,
    generate_quest_metadata,
    get_player_profile,
    recommend_next_action,
    validate_quest_completion,
)


def _check_inventory_tool(player_id: str, item_name: str) -> dict:
    profile = get_player_profile(player_id)
    return check_inventory(profile, item_name)


def _recommend_next_action_tool(player_id: str) -> dict:
    profile = get_player_profile(player_id)
    return {
        "player_id": player_id,
        "recommendation": recommend_next_action(profile),
    }


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_player_profile": get_player_profile,
    "check_inventory": _check_inventory_tool,
    "recommend_next_action": _recommend_next_action_tool,
    "generate_quest": generate_quest_metadata,
    "validate_quest_completion": validate_quest_completion,
}


def execute_tool_call(tool_call: dict) -> dict:
    tool_name = tool_call.get("name")
    tool_input = tool_call.get("input") or {}

    if not isinstance(tool_input, dict):
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": "tool input must be an object",
        }

    if tool_name not in TOOL_FUNCTIONS:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"unknown tool: {tool_name}",
        }

    function = TOOL_FUNCTIONS[tool_name]
    signature = inspect.signature(function)
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and name not in tool_input
    ]
    if missing:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"missing required argument: {missing[0]}",
        }
    filtered_input = {
        name: tool_input[name]
        for name in signature.parameters
        if name in tool_input
    }

    try:
        result = function(**filtered_input)
    except Exception as exc:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": str(exc),
        }

    return {
        "ok": True,
        "tool_name": tool_name,
        "result": result,
    }


def tool_result_block(tool_use_id: str, execution_result: dict) -> dict:
    block = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(execution_result, ensure_ascii=False),
    }
    if execution_result.get("ok") is False:
        block["is_error"] = True
    return block
