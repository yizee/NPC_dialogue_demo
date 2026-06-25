from memory_store import DialogueMemorySession
from npc_dialogue import (
    compact_memory_if_needed,
    extract_inventory_item_name,
    generate_gm_feedback_safely,
    get_live_conversation_history,
    is_gm_advisor,
    record_memory_turn,
    should_use_rag_for_intent,
)


def test_should_use_rag_only_for_lore_question():
    assert should_use_rag_for_intent("lore_question") is True
    assert should_use_rag_for_intent("persona_chat") is False
    assert should_use_rag_for_intent("inventory_query") is False
    assert should_use_rag_for_intent("quest_request") is False
    assert should_use_rag_for_intent("gm_feedback") is False


def test_is_gm_advisor_only_for_feedback_npc():
    assert is_gm_advisor("gm_advisor") is True
    assert is_gm_advisor("blacksmith") is False


def test_extract_inventory_item_name_removes_common_question_words():
    assert extract_inventory_item_name("我背包里有月盐药剂吗？") == "月盐药剂"
    assert extract_inventory_item_name("Do I have the black iron tongs in my inventory?") == "black iron tongs"


def test_extract_inventory_item_name_keeps_direct_item_name():
    assert extract_inventory_item_name("魔龙之心") == "魔龙之心"


def test_record_memory_turn_appends_exchange(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
    )

    record_memory_turn(
        memory_session=session,
        user_input="你好",
        assistant_reply="铁砧不会等人。",
        intent="persona_chat",
        used_rag=False,
        used_tool_calling=False,
    )

    assert session.turn_count == 1
    assert session.recent_conversation_history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "铁砧不会等人。"},
    ]
    assert session.raw_log_path.exists()


def test_get_live_conversation_history_uses_recent_memory_only(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
        max_recent_messages=2,
    )
    session.append_turn("old user", "old assistant", "persona_chat")
    session.append_turn("new user", "new assistant", "persona_chat")

    assert get_live_conversation_history(session) == [
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ]


def test_compact_memory_if_needed_keeps_existing_summary_without_api_call(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
    )
    session.session_summary = "旧摘要"

    compact_memory_if_needed(session, force=True)

    assert session.session_summary == "旧摘要"


def test_generate_gm_feedback_safely_returns_fallback_when_api_fails(monkeypatch):
    def failing_feedback(npc, player_context, conversation_history):
        raise RuntimeError("missing api key")

    monkeypatch.setattr("npc_dialogue.generate_gm_feedback", failing_feedback)

    feedback = generate_gm_feedback_safely(
        npc={"name": "格林大叔"},
        player_context={"name": "旅行者"},
        conversation_history=[{"role": "user", "content": "你好"}],
    )

    assert "自动 GM 反馈生成失败" in feedback
    assert "missing api key" in feedback
