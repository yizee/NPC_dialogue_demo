import pytest

from intent_router import classify_intent


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我要反馈这个任务不合理", "gm_feedback"),
        ("This quest is bugged and I want to report it", "gm_feedback"),
        ("我背包里有月盐药剂吗？", "inventory_query"),
        ("Do I have the black iron tongs in my inventory?", "inventory_query"),
        ("我想接一个任务", "quest_request"),
        ("Do you have a mission for me?", "quest_request"),
        ("碎星髓是什么？", "lore_question"),
        ("Who is the Goblin King?", "lore_question"),
        ("你今天怎么样？", "persona_chat"),
        ("hello friend", "persona_chat"),
    ],
)
def test_classify_intent_examples(text, expected):
    result = classify_intent(text)

    assert result.intent == expected
    assert 0.0 <= result.confidence <= 1.0
    assert result.reason


def test_intent_priority_prefers_feedback_over_quest():
    result = classify_intent("我要反馈这个任务不合理")

    assert result.intent == "gm_feedback"


def test_intent_priority_prefers_inventory_over_lore_item_words():
    result = classify_intent("我有没有魔龙之心这个道具？")

    assert result.intent == "inventory_query"


def test_empty_input_defaults_to_persona_chat_with_zero_confidence():
    result = classify_intent("  ")

    assert result.intent == "persona_chat"
    assert result.confidence == 0.0
    assert result.reason == "empty input"
