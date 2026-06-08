from __future__ import annotations

from dataclasses import dataclass


PERSONA_CHAT = "persona_chat"
LORE_QUESTION = "lore_question"
QUEST_REQUEST = "quest_request"
INVENTORY_QUERY = "inventory_query"
GM_FEEDBACK = "gm_feedback"

VALID_INTENTS = {
    PERSONA_CHAT,
    LORE_QUESTION,
    QUEST_REQUEST,
    INVENTORY_QUERY,
    GM_FEEDBACK,
}


@dataclass(frozen=True)
class RouteResult:
    intent: str
    confidence: float
    reason: str


KEYWORDS_BY_INTENT = {
    GM_FEEDBACK: [
        "投诉",
        "反馈",
        "举报",
        "卡住",
        "不合理",
        "bug",
        "bugged",
        "complaint",
        "feedback",
        "report",
        "stuck",
        "broken",
    ],
    INVENTORY_QUERY: [
        "背包",
        "物品",
        "道具",
        "有没有",
        "是否拥有",
        "我有",
        "inventory",
        "backpack",
        "item",
        "do i have",
        "have the",
    ],
    QUEST_REQUEST: [
        "接任务",
        "任务",
        "委托",
        "有什么事要我做",
        "给我一个任务",
        "quest",
        "mission",
        "task",
        "something to do",
    ],
    LORE_QUESTION: [
        "是什么",
        "谁是",
        "在哪里",
        "为什么",
        "世界观",
        "设定",
        "背景",
        "魔龙",
        "地精之王",
        "碎星髓",
        "真结局",
        "who is",
        "what is",
        "where is",
        "why",
        "lore",
        "world",
        "monster",
        "goblin king",
        "dragon",
    ],
}


def _matched_keyword(text: str, keywords: list[str]) -> str | None:
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() in lowered:
            return keyword
    return None


def classify_intent(player_input: str) -> RouteResult:
    text = player_input.strip()
    if not text:
        return RouteResult(PERSONA_CHAT, 0.0, "empty input")

    priority = [GM_FEEDBACK, INVENTORY_QUERY, QUEST_REQUEST, LORE_QUESTION]
    for intent in priority:
        matched = _matched_keyword(text, KEYWORDS_BY_INTENT[intent])
        if matched:
            return RouteResult(intent, 0.9, f"matched keyword: {matched}")

    return RouteResult(PERSONA_CHAT, 0.5, "no workflow keyword matched")
