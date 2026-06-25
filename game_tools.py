from __future__ import annotations


DEFAULT_PLAYER_PROFILE = {
    "player_id": "player_001",
    "name": "旅行者",
    "level": 5,
    "class": "未选择",
    "inventory": ["黑铁夹具", "月盐药剂"],
    "active_quest": None,
}

ITEM_ALIASES = {
    "black iron tongs": "黑铁夹具",
    "moon salt potion": "月盐药剂",
}

QUEST_REQUIREMENTS = {
    "dragon_hunter_weapon": ["裂鳞长枪"],
    "moon_salt_potion": ["月盐药剂"],
    "star_marrow_sample": ["碎星髓"],
}


def get_player_profile(player_id: str) -> dict:
    profile = dict(DEFAULT_PLAYER_PROFILE)
    profile["player_id"] = player_id
    profile["inventory"] = list(DEFAULT_PLAYER_PROFILE["inventory"])
    return profile


def check_inventory(player_context: dict, item_name: str) -> dict:
    inventory = player_context.get("inventory", [])
    normalized_query = item_name.strip()
    canonical_item_name = ITEM_ALIASES.get(normalized_query.lower(), normalized_query)
    has_item = bool(normalized_query) and any(
        canonical_item_name in item or item in canonical_item_name
        for item in inventory
    )
    return {
        "item_name": canonical_item_name,
        "has_item": has_item,
        "inventory": list(inventory),
    }


def format_inventory_response(result: dict) -> str:
    item_name = result["item_name"]
    if result["has_item"]:
        return f"[工具调用: check_inventory] 你有{item_name}。"
    return f"[工具调用: check_inventory] 你还没有{item_name}。"


def recommend_next_action(player_state: dict) -> str:
    active_quest = player_state.get("active_quest")
    if active_quest:
        return f"建议继续推进当前任务：{active_quest}。"

    if player_state.get("class", "未选择") == "未选择":
        return "建议先选择一个主要职业：可以找艾尔文走法师路线，找格林大叔走战士路线，或找罗莎老板娘走游侠路线。"

    if player_state.get("level", 1) < 5:
        return "建议先完成外围调查或采集任务提升等级。"

    return "建议调查魔龙猎人遗物，为挑战灰烬魔龙做准备。"


def generate_quest_metadata(npc_id: str, player_level: int) -> dict:
    quest_type_by_npc = {
        "blacksmith": "收集魔龙猎人武器",
        "innkeeper": "调查失踪旅客情报",
        "mysterious_wizard": "收集安全碎星髓样本",
    }
    difficulty = "low"
    if player_level >= 8:
        difficulty = "high"
    elif player_level >= 5:
        difficulty = "medium"

    return {
        "npc_id": npc_id,
        "player_level": player_level,
        "recommended_type": quest_type_by_npc.get(npc_id, "外围调查"),
        "difficulty": difficulty,
    }


def validate_quest_completion(
    player_id: str,
    quest_id: str,
    evidence: list[str] | None = None,
) -> dict:
    required_evidence = QUEST_REQUIREMENTS.get(quest_id, [])
    provided_evidence = evidence or []
    completed = bool(required_evidence) and all(
        required in provided_evidence for required in required_evidence
    )
    return {
        "player_id": player_id,
        "quest_id": quest_id,
        "completed": completed,
        "required_evidence": list(required_evidence),
        "provided_evidence": list(provided_evidence),
    }
