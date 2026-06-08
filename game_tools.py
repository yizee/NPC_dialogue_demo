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
