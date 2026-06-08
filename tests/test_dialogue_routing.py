from npc_dialogue import extract_inventory_item_name, should_use_rag_for_intent


def test_should_use_rag_only_for_lore_question():
    assert should_use_rag_for_intent("lore_question") is True
    assert should_use_rag_for_intent("persona_chat") is False
    assert should_use_rag_for_intent("inventory_query") is False
    assert should_use_rag_for_intent("quest_request") is False
    assert should_use_rag_for_intent("gm_feedback") is False


def test_extract_inventory_item_name_removes_common_question_words():
    assert extract_inventory_item_name("我背包里有月盐药剂吗？") == "月盐药剂"
    assert extract_inventory_item_name("Do I have the black iron tongs in my inventory?") == "black iron tongs"


def test_extract_inventory_item_name_keeps_direct_item_name():
    assert extract_inventory_item_name("魔龙之心") == "魔龙之心"
