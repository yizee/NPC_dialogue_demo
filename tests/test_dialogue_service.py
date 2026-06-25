from dialogue_service import DialogueWebService


def test_list_npcs_returns_existing_characters():
    service = DialogueWebService(log_dir="logs/test_web_sessions")

    npcs = service.list_npcs()

    npc_ids = {npc["id"] for npc in npcs}
    assert {"blacksmith", "innkeeper", "mysterious_wizard", "gm_advisor"} <= npc_ids
    assert all("name" in npc for npc in npcs)
    assert all("title" in npc for npc in npcs)
    assert all("greeting" in npc for npc in npcs)


def test_create_session_returns_npc_and_mock_mode(tmp_path):
    service = DialogueWebService(log_dir=tmp_path)

    response = service.create_session(
        npc_id="blacksmith",
        player_id="demo_player",
        mode="mock",
    )

    assert response["mode"] == "mock"
    assert response["npc"]["id"] == "blacksmith"
    assert response["npc"]["name"] == "格林大叔"
    assert response["session_id"] in service.sessions


def test_create_session_restores_previous_memory_for_same_npc_and_player(tmp_path):
    first_service = DialogueWebService(log_dir=tmp_path)
    first_session = first_service.create_session("blacksmith", "demo_player", "mock")
    first_service.chat(
        session_id=first_session["session_id"],
        npc_id="blacksmith",
        player_id="demo_player",
        message="我叫林恩，请记住。",
        mode="mock",
    )

    second_service = DialogueWebService(log_dir=tmp_path)
    second_session = second_service.create_session("blacksmith", "demo_player", "mock")
    memory_session = second_service.sessions[second_session["session_id"]]

    assert memory_session.recent_conversation_history[0] == {
        "role": "user",
        "content": "我叫林恩，请记住。",
    }
    assert "林恩" in memory_session.recent_conversation_history[1]["content"]


def test_mock_chat_classifies_intent_and_records_memory(tmp_path):
    service = DialogueWebService(log_dir=tmp_path)
    session = service.create_session("blacksmith", "demo_player", "mock")

    response = service.chat(
        session_id=session["session_id"],
        npc_id="blacksmith",
        player_id="demo_player",
        message="我背包里有没有月盐药剂？",
        mode="mock",
    )

    assert "Mock" in response["reply"]
    assert response["metadata"]["intent"] == "inventory_query"
    assert response["metadata"]["tool_used"] is True
    assert response["metadata"]["rag_used"] is False
    assert response["metadata"]["memory_recorded"] is True
    memory_session = service.sessions[session["session_id"]]
    assert memory_session.turn_count == 1
    assert memory_session.raw_log_path.exists()


def test_lore_question_sets_rag_metadata_in_mock_mode(tmp_path):
    service = DialogueWebService(log_dir=tmp_path)
    session = service.create_session("mysterious_wizard", "demo_player", "mock")

    response = service.chat(
        session_id=session["session_id"],
        npc_id="mysterious_wizard",
        player_id="demo_player",
        message="碎星髓是什么？",
        mode="mock",
    )

    assert response["metadata"]["intent"] == "lore_question"
    assert response["metadata"]["rag_used"] is True
    assert response["metadata"]["tool_used"] is False


def test_unknown_npc_raises_clear_value_error(tmp_path):
    service = DialogueWebService(log_dir=tmp_path)

    try:
        service.create_session("missing_npc", "demo_player", "mock")
    except ValueError as exc:
        assert "Unknown NPC ID" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_empty_message_raises_clear_value_error(tmp_path):
    service = DialogueWebService(log_dir=tmp_path)
    session = service.create_session("blacksmith", "demo_player", "mock")

    try:
        service.chat(
            session_id=session["session_id"],
            npc_id="blacksmith",
            player_id="demo_player",
            message="   ",
            mode="mock",
        )
    except ValueError as exc:
        assert "Message cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_live_mode_without_api_key_returns_mock_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = DialogueWebService(log_dir=tmp_path)
    session = service.create_session("blacksmith", "demo_player", "live")

    response = service.chat(
        session_id=session["session_id"],
        npc_id="blacksmith",
        player_id="demo_player",
        message="你好",
        mode="live",
    )

    assert response["metadata"]["mode"] == "mock"
    assert response["metadata"]["requested_mode"] == "live"
    assert response["metadata"]["fallback"] == "ANTHROPIC_API_KEY is not configured"


def test_live_mode_with_api_key_uses_live_persona_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def fake_create_claude_message(self, *, system_prompt, messages, max_tokens=300):
        calls.append({
            "system_prompt": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens,
        })
        return "真实 NPC 回复"

    monkeypatch.setattr(
        DialogueWebService,
        "_create_claude_message",
        fake_create_claude_message,
        raising=False,
    )
    service = DialogueWebService(log_dir=tmp_path)
    session = service.create_session("blacksmith", "demo_player", "live")

    response = service.chat(
        session_id=session["session_id"],
        npc_id="blacksmith",
        player_id="demo_player",
        message="你好",
        mode="live",
    )

    assert response["reply"] == "真实 NPC 回复"
    assert calls
    assert calls[0]["messages"][-1] == {"role": "user", "content": "你好"}
    assert response["metadata"]["mode"] == "live"
    assert response["metadata"]["requested_mode"] == "live"
    assert response["metadata"]["fallback"] is None
    assert response["metadata"]["intent"] == "persona_chat"
    assert response["metadata"]["memory_recorded"] is True
