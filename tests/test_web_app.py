from fastapi.testclient import TestClient

from dialogue_service import DialogueWebService
from web_app import create_app


def make_client(tmp_path):
    service = DialogueWebService(log_dir=tmp_path)
    app = create_app(service=service)
    return TestClient(app)


def test_get_npcs_returns_configured_npcs(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/api/npcs")

    assert response.status_code == 200
    npc_ids = {npc["id"] for npc in response.json()["npcs"]}
    assert "blacksmith" in npc_ids
    assert "gm_advisor" in npc_ids


def test_create_session_returns_session_payload(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/api/sessions",
        json={"npc_id": "innkeeper", "player_id": "demo_player", "mode": "mock"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["npc"]["id"] == "innkeeper"
    assert body["mode"] == "mock"
    assert body["session_id"]


def test_chat_returns_reply_and_metadata(tmp_path):
    client = make_client(tmp_path)
    session = client.post(
        "/api/sessions",
        json={"npc_id": "blacksmith", "player_id": "demo_player", "mode": "mock"},
    ).json()

    response = client.post(
        "/api/chat",
        json={
            "session_id": session["session_id"],
            "npc_id": "blacksmith",
            "player_id": "demo_player",
            "message": "我背包里有没有月盐药剂？",
            "mode": "mock",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "reply" in body
    assert body["metadata"]["intent"] == "inventory_query"
    assert body["metadata"]["tool_used"] is True


def test_unknown_npc_returns_404(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/api/sessions",
        json={"npc_id": "missing_npc", "player_id": "demo_player", "mode": "mock"},
    )

    assert response.status_code == 404
    assert "Unknown NPC ID" in response.json()["detail"]


def test_empty_chat_message_returns_400(tmp_path):
    client = make_client(tmp_path)
    session = client.post(
        "/api/sessions",
        json={"npc_id": "blacksmith", "player_id": "demo_player", "mode": "mock"},
    ).json()

    response = client.post(
        "/api/chat",
        json={
            "session_id": session["session_id"],
            "npc_id": "blacksmith",
            "player_id": "demo_player",
            "message": " ",
            "mode": "mock",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty"


def test_index_page_is_served(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "NPC Dialogue Console" in response.text
