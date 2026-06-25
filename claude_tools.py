from __future__ import annotations


CLAUDE_TOOLS = [
    {
        "name": "get_player_profile",
        "description": "Read the player's current level, class, inventory, and active quest.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string", "description": "Player identifier."},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "check_inventory",
        "description": "Check whether the player owns a specific item.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "item_name": {"type": "string"},
            },
            "required": ["player_id", "item_name"],
        },
    },
    {
        "name": "recommend_next_action",
        "description": "Recommend what the player should do next based on player state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "generate_quest",
        "description": "Generate quest metadata based on NPC and player level. This does not replace the narrative quest generator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "npc_id": {"type": "string"},
                "player_level": {"type": "integer"},
            },
            "required": ["npc_id", "player_level"],
        },
    },
    {
        "name": "validate_quest_completion",
        "description": "Validate whether a quest is completed using explicit evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "quest_id": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence items the player provides.",
                },
            },
            "required": ["player_id", "quest_id"],
        },
    },
]


def get_tool_schema(tool_name: str) -> dict | None:
    for tool in CLAUDE_TOOLS:
        if tool["name"] == tool_name:
            return tool
    return None
