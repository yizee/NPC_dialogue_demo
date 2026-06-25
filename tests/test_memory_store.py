import json

from memory_store import DialogueMemorySession, create_session_id


def test_create_session_id_contains_npc_and_player():
    session_id = create_session_id("mysterious_wizard", "player_001")

    assert "mysterious_wizard" in session_id
    assert "player_001" in session_id


def test_append_turn_writes_jsonl(tmp_path):
    session = DialogueMemorySession(
        npc_id="mysterious_wizard",
        player_id="player_001",
        log_dir=tmp_path,
    )

    session.append_turn(
        user_input="我背包里有月盐药剂吗？",
        assistant_reply="你确实带着一瓶月盐药剂。",
        intent="inventory_query",
        used_rag=False,
        used_tool_calling=True,
        rag_sources=[],
        tool_names=["check_inventory"],
    )

    lines = session.raw_log_path.read_text(encoding="utf-8").splitlines()
    saved = json.loads(lines[0])

    assert saved["session_id"] == session.session_id
    assert saved["turn_index"] == 1
    assert saved["npc_id"] == "mysterious_wizard"
    assert saved["player_id"] == "player_001"
    assert saved["user_input"] == "我背包里有月盐药剂吗？"
    assert saved["assistant_reply"] == "你确实带着一瓶月盐药剂。"
    assert saved["intent"] == "inventory_query"
    assert saved["used_rag"] is False
    assert saved["used_tool_calling"] is True
    assert saved["tool_names"] == ["check_inventory"]


def test_recent_history_is_trimmed_to_max_messages(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
        max_recent_messages=4,
    )

    for index in range(3):
        session.append_turn(
            user_input=f"user {index}",
            assistant_reply=f"assistant {index}",
            intent="persona_chat",
        )

    assert session.recent_conversation_history == [
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
        {"role": "user", "content": "user 2"},
        {"role": "assistant", "content": "assistant 2"},
    ]
    assert session.full_conversation_history == [
        {"role": "user", "content": "user 0"},
        {"role": "assistant", "content": "assistant 0"},
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
        {"role": "user", "content": "user 2"},
        {"role": "assistant", "content": "assistant 2"},
    ]


def test_should_compact_by_recent_message_count(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
        max_recent_messages=4,
    )

    for index in range(3):
        session.append_turn(
            user_input=f"user {index}",
            assistant_reply=f"assistant {index}",
            intent="persona_chat",
        )

    assert session.should_compact() is True
    assert len(session.recent_conversation_history) == 4

    session.mark_compacted()

    assert session.should_compact() is False


def test_should_compact_by_turn_interval(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
        compact_every_n_turns=2,
    )

    session.turn_count = 2

    assert session.should_compact() is True


def test_story_events_and_quest_state_are_updated_separately(tmp_path):
    session = DialogueMemorySession(
        npc_id="mysterious_wizard",
        player_id="player_001",
        log_dir=tmp_path,
    )

    session.add_story_event("clue_revealed", "神秘法师暗示旧塔藏有入门魔法书。")
    session.update_quest_state(
        active_quest="wizard_starter_1",
        known_clues=["旧塔可能藏有入门魔法书"],
        pending_commitments=["玩家承诺寻找入门魔法书"],
    )

    assert session.story_events == [{
        "event_type": "clue_revealed",
        "summary": "神秘法师暗示旧塔藏有入门魔法书。",
    }]
    assert session.quest_state["active_quest"] == "wizard_starter_1"
    assert session.quest_state["known_clues"] == ["旧塔可能藏有入门魔法书"]
    assert session.quest_state["pending_commitments"] == ["玩家承诺寻找入门魔法书"]


def test_build_compact_input_contains_summary_recent_events_and_quest_state(tmp_path):
    session = DialogueMemorySession(
        npc_id="mysterious_wizard",
        player_id="player_001",
        log_dir=tmp_path,
    )
    session.session_summary = "玩家与神秘法师初步建立信任。"
    session.recent_conversation_history = [
        {"role": "user", "content": "我想学习魔法。"},
        {"role": "assistant", "content": "先证明你的耐心。"},
    ]
    session.add_story_event("clue_revealed", "旧塔可能藏有入门魔法书。")
    session.update_quest_state(active_quest="wizard_starter_1")

    compact_input = session.build_compact_input()

    assert compact_input["session_summary"] == "玩家与神秘法师初步建立信任。"
    assert compact_input["full_conversation_history"] == session.full_conversation_history
    assert compact_input["recent_conversation_history"] == session.recent_conversation_history
    assert compact_input["story_events"] == session.story_events
    assert compact_input["quest_state"]["active_quest"] == "wizard_starter_1"
