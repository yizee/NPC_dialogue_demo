from pathlib import Path


STATIC_DIR = Path("web_static")


def test_index_contains_required_ui_hooks():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    required_ids = [
        "npcList",
        "modeSelect",
        "chatLog",
        "messageInput",
        "sendButton",
        "intentStatus",
        "ragStatus",
        "toolStatus",
        "memoryStatus",
        "fallbackStatus",
    ]
    for element_id in required_ids:
        assert f'id="{element_id}"' in html


def test_app_js_references_required_api_endpoints():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/npcs")' in js
    assert 'fetch("/api/sessions"' in js
    assert 'fetch("/api/chat"' in js


def test_styles_define_message_bubbles_and_status_panel():
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert ".message.player" in css
    assert ".message.npc" in css
    assert ".status-grid" in css
