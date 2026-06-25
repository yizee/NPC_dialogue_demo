# Local Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI web chat interface for GameNPCDialogue that works in mock mode without credentials and exposes a guarded live-mode boundary for the existing Claude/RAG/tool/memory pipeline.

**Architecture:** Add a focused `dialogue_service.py` layer that wraps NPC config loading, intent routing, deterministic mock replies, live-mode fallback, and memory logging. Add `web_app.py` as the FastAPI boundary and serve plain static files from `web_static/`. Keep CLI behavior in `npc_dialogue.py` intact for this first implementation.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, Starlette TestClient, pytest, vanilla HTML/CSS/JS.

---

## File Structure

- Create `dialogue_service.py`: reusable service for web sessions, mock replies, metadata, and live fallback.
- Create `web_app.py`: FastAPI app factory, JSON endpoints, static file serving, and local server entrypoint.
- Create `web_static/index.html`: local chat UI markup.
- Create `web_static/styles.css`: compact portfolio-ready UI styling.
- Create `web_static/app.js`: browser state, API calls, transcript rendering, and status panel updates.
- Create `tests/test_dialogue_service.py`: unit tests for NPC listing, session creation, mock chat, memory logging, and live fallback.
- Create `tests/test_web_app.py`: API tests for `/api/npcs`, `/api/sessions`, `/api/chat`, validation errors, and static page serving.
- Create `tests/test_web_static.py`: static asset smoke tests for required DOM hooks and endpoint references.
- Modify `requirements.txt`: add FastAPI, Uvicorn, and HTTPX/TestClient-compatible dependency.
- Modify `README.md`: document local web demo setup and run commands.

---

### Task 1: Add FastAPI Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update dependency list**

Add these lines to `requirements.txt` after the existing LangChain dependencies and before `pytest`:

```text
fastapi>=0.116,<1
uvicorn[standard]>=0.35,<1
httpx>=0.28,<1
```

- [ ] **Step 2: Install dependencies**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pip install -r requirements.txt
```

Expected: pip completes successfully and installs FastAPI, Uvicorn, and HTTPX if they are not already present.

- [ ] **Step 3: Run dependency import smoke check**

Run:

```bash
/Applications/anaconda3/bin/python3.13 - <<'PY'
import fastapi
import uvicorn
import httpx
print("web dependencies import OK")
PY
```

Expected:

```text
web dependencies import OK
```

- [ ] **Step 4: Commit dependency change**

Run:

```bash
git add requirements.txt
git commit -m "chore: add web demo dependencies"
```

Expected: a commit that changes only `requirements.txt`.

---

### Task 2: Create Dialogue Service With Mock Mode

**Files:**
- Create: `dialogue_service.py`
- Create: `tests/test_dialogue_service.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/test_dialogue_service.py`:

```python
import os

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest tests/test_dialogue_service.py -v
```

Expected: FAIL because `dialogue_service.py` does not exist yet.

- [ ] **Step 3: Implement `dialogue_service.py`**

Create `dialogue_service.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from intent_router import (
    GM_FEEDBACK,
    INVENTORY_QUERY,
    LORE_QUESTION,
    PERSONA_CHAT,
    QUEST_REQUEST,
    classify_intent,
)
from memory_store import DialogueMemorySession
from npc_dialogue import load_all_npc_configs, load_npc_config


VALID_MODES = {"mock", "live"}


class DialogueWebService:
    def __init__(self, log_dir: str | Path = "logs/web_sessions") -> None:
        self.log_dir = Path(log_dir)
        self.sessions: dict[str, DialogueMemorySession] = {}

    def list_npcs(self) -> list[dict[str, Any]]:
        configs = load_all_npc_configs()
        return [
            self._public_npc_payload(npc_id, npc)
            for npc_id, npc in configs.items()
        ]

    def create_session(
        self,
        npc_id: str,
        player_id: str = "demo_player",
        mode: str = "mock",
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        npc = self._load_npc(npc_id)
        session = DialogueMemorySession(
            npc_id=npc_id,
            player_id=player_id,
            log_dir=self.log_dir,
        )
        self.sessions[session.session_id] = session
        return {
            "session_id": session.session_id,
            "npc": self._public_npc_payload(npc_id, npc),
            "mode": mode,
        }

    def chat(
        self,
        session_id: str,
        npc_id: str,
        player_id: str = "demo_player",
        message: str = "",
        mode: str = "mock",
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        if not message.strip():
            raise ValueError("Message cannot be empty")
        npc = self._load_npc(npc_id)
        session = self._get_or_create_session(session_id, npc_id, player_id)

        requested_mode = mode
        fallback = None
        if mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
            mode = "mock"
            fallback = "ANTHROPIC_API_KEY is not configured"

        route = classify_intent(message)
        reply = self._mock_reply(npc=npc, intent=route.intent, message=message)
        rag_used = route.intent == LORE_QUESTION
        tool_used = route.intent == INVENTORY_QUERY

        memory_recorded = False
        try:
            session.append_turn(
                user_input=message,
                assistant_reply=reply,
                intent=route.intent,
                used_rag=rag_used,
                used_tool_calling=tool_used,
                rag_sources=["mock_lore_context"] if rag_used else [],
                tool_names=["mock_inventory_check"] if tool_used else [],
            )
            memory_recorded = True
        except OSError:
            memory_recorded = False

        return {
            "reply": reply,
            "metadata": {
                "intent": route.intent,
                "intent_reason": route.reason,
                "rag_used": rag_used,
                "tool_used": tool_used,
                "memory_recorded": memory_recorded,
                "session_id": session.session_id,
                "turn_count": session.turn_count,
                "mode": mode,
                "requested_mode": requested_mode,
                "fallback": fallback,
            },
        }

    def _load_npc(self, npc_id: str) -> dict[str, Any]:
        try:
            return load_npc_config(npc_id)
        except ValueError as exc:
            raise ValueError(f"Unknown NPC ID: {npc_id}") from exc

    def _get_or_create_session(
        self,
        session_id: str,
        npc_id: str,
        player_id: str,
    ) -> DialogueMemorySession:
        if session_id in self.sessions:
            return self.sessions[session_id]
        session = DialogueMemorySession(
            npc_id=npc_id,
            player_id=player_id,
            log_dir=self.log_dir,
            session_id=session_id or None,
        )
        self.sessions[session.session_id] = session
        return session

    def _public_npc_payload(self, npc_id: str, npc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": npc_id,
            "name": npc["name"],
            "title": npc["title"],
            "greeting": npc["greeting"],
            "personality": npc.get("personality", ""),
            "topics": npc.get("topics", []),
        }

    def _validate_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")

    def _mock_reply(self, npc: dict[str, Any], intent: str, message: str) -> str:
        name = npc["name"]
        if intent == INVENTORY_QUERY:
            return f"[Mock] {name}: 我先查了一下你的背包记录。这个演示分支会标记为 tool_used，真实模式会调用工具层。"
        if intent == QUEST_REQUEST:
            return f"[Mock] {name}: 我可以给你一个小任务：沿着镇北的小路调查异常线索，再回来告诉我发现。"
        if intent == LORE_QUESTION:
            return f"[Mock] {name}: 这个问题会触发 lore/RAG 路径。当前 mock 模式先用本地占位知识回答：{message}"
        if intent == GM_FEEDBACK:
            return f"[Mock] {name}: 我会把这条反馈记录为 GM review 输入，并指出任务目标、触发条件和奖励说明是否清楚。"
        if intent == PERSONA_CHAT:
            return f"[Mock] {name}: {npc['greeting']} 你刚才说的是：{message}"
        return f"[Mock] {name}: 我听到了。"
```

- [ ] **Step 4: Run service tests**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest tests/test_dialogue_service.py -v
```

Expected: all tests in `tests/test_dialogue_service.py` pass.

- [ ] **Step 5: Commit service layer**

Run:

```bash
git add dialogue_service.py tests/test_dialogue_service.py
git commit -m "feat: add mock dialogue web service"
```

Expected: a commit containing only the service and service tests.

---

### Task 3: Add FastAPI App and API Tests

**Files:**
- Create: `web_app.py`
- Create: `tests/test_web_app.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_web_app.py`:

```python
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
```

- [ ] **Step 2: Run API tests to verify they fail**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest tests/test_web_app.py -v
```

Expected: FAIL because `web_app.py` and static files do not exist yet.

- [ ] **Step 3: Implement `web_app.py`**

Create `web_app.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dialogue_service import DialogueWebService


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"


class SessionRequest(BaseModel):
    npc_id: str
    player_id: str = "demo_player"
    mode: Literal["mock", "live"] = "mock"


class ChatRequest(BaseModel):
    session_id: str
    npc_id: str
    player_id: str = "demo_player"
    message: str
    mode: Literal["mock", "live"] = "mock"


def create_app(service: DialogueWebService | None = None) -> FastAPI:
    app = FastAPI(title="GameNPCDialogue Web Demo")
    dialogue_service = service or DialogueWebService()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="web_static/index.html not found")
        return FileResponse(index_path)

    @app.get("/api/npcs")
    def get_npcs():
        return {"npcs": dialogue_service.list_npcs()}

    @app.post("/api/sessions")
    def create_session(request: SessionRequest):
        try:
            return dialogue_service.create_session(
                npc_id=request.npc_id,
                player_id=request.player_id,
                mode=request.mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        try:
            return dialogue_service.chat(
                session_id=request.session_id,
                npc_id=request.npc_id,
                player_id=request.player_id,
                message=request.message,
                mode=request.mode,
            )
        except ValueError as exc:
            message = str(exc)
            status_code = 400 if "Message cannot be empty" in message else 404
            raise HTTPException(status_code=status_code, detail=message) from exc

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
```

- [ ] **Step 4: Add temporary minimal `web_static/index.html`**

Create `web_static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NPC Dialogue Console</title>
</head>
<body>
  <main>
    <h1>NPC Dialogue Console</h1>
    <p>Local web demo shell.</p>
  </main>
</body>
</html>
```

- [ ] **Step 5: Run API tests**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest tests/test_web_app.py -v
```

Expected: all tests in `tests/test_web_app.py` pass.

- [ ] **Step 6: Commit API layer**

Run:

```bash
git add web_app.py web_static/index.html tests/test_web_app.py
git commit -m "feat: add local web api"
```

Expected: a commit containing the FastAPI app, minimal static page, and API tests.

---

### Task 4: Build Static Web Chat UI

**Files:**
- Modify: `web_static/index.html`
- Create: `web_static/styles.css`
- Create: `web_static/app.js`
- Create: `tests/test_web_static.py`

- [ ] **Step 1: Write static asset smoke tests**

Create `tests/test_web_static.py`:

```python
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
```

- [ ] **Step 2: Run static tests to verify they fail**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest tests/test_web_static.py -v
```

Expected: FAIL because `app.js`, `styles.css`, and required DOM hooks are not complete.

- [ ] **Step 3: Replace `web_static/index.html` with the full UI**

Replace `web_static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NPC Dialogue Console</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="app-shell">
    <aside class="npc-panel">
      <div class="brand">
        <span class="brand-mark">NPC</span>
        <div>
          <h1>NPC Dialogue Console</h1>
          <p>Local agent workflow demo</p>
        </div>
      </div>

      <div class="control-block">
        <label for="modeSelect">Mode</label>
        <select id="modeSelect">
          <option value="mock">Mock</option>
          <option value="live">Live</option>
        </select>
      </div>

      <div class="control-block">
        <div class="section-label">Characters</div>
        <div id="npcList" class="npc-list"></div>
      </div>
    </aside>

    <section class="chat-panel">
      <header class="character-header">
        <div>
          <div id="npcName" class="npc-name">Loading...</div>
          <div id="npcTitle" class="npc-title"></div>
        </div>
        <div id="sessionBadge" class="session-badge">No session</div>
      </header>

      <div id="npcPersona" class="persona"></div>
      <div id="chatLog" class="chat-log"></div>

      <form id="chatForm" class="chat-form">
        <input id="messageInput" type="text" autocomplete="off" placeholder="Ask about lore, inventory, quests, or feedback">
        <button id="sendButton" type="submit">Send</button>
      </form>
      <div id="errorBox" class="error-box" hidden></div>
    </section>

    <aside class="status-panel">
      <div class="section-label">System State</div>
      <div class="status-grid">
        <div class="status-item">
          <span>Intent</span>
          <strong id="intentStatus">-</strong>
        </div>
        <div class="status-item">
          <span>RAG</span>
          <strong id="ragStatus">-</strong>
        </div>
        <div class="status-item">
          <span>Tool</span>
          <strong id="toolStatus">-</strong>
        </div>
        <div class="status-item">
          <span>Memory</span>
          <strong id="memoryStatus">-</strong>
        </div>
      </div>
      <div class="fallback-box">
        <span>Fallback</span>
        <p id="fallbackStatus">None</p>
      </div>
    </aside>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Create `web_static/styles.css`**

Create `web_static/styles.css`:

```css
:root {
  color-scheme: light;
  --bg: #f5f7f2;
  --panel: #ffffff;
  --ink: #1f2623;
  --muted: #66736f;
  --line: #d9e0dc;
  --accent: #1f7a5b;
  --accent-dark: #145640;
  --npc: #eef4ff;
  --player: #e8f6ef;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
}

.app-shell {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 240px;
  gap: 16px;
  min-height: 100vh;
  padding: 16px;
}

.npc-panel,
.chat-panel,
.status-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.npc-panel,
.status-panel {
  padding: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  font-weight: 700;
}

h1 {
  margin: 0;
  font-size: 18px;
}

p {
  margin: 0;
}

.brand p,
.npc-title,
.persona,
.section-label,
.status-item span,
.fallback-box span {
  color: var(--muted);
}

.control-block {
  margin-bottom: 18px;
}

label,
.section-label {
  display: block;
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0;
}

select,
input,
button {
  width: 100%;
  min-height: 40px;
  border: 1px solid var(--line);
  border-radius: 6px;
  font: inherit;
}

select,
input {
  padding: 0 12px;
  background: white;
  color: var(--ink);
}

button {
  padding: 0 14px;
  border-color: var(--accent);
  background: var(--accent);
  color: white;
  font-weight: 700;
  cursor: pointer;
}

button:hover {
  background: var(--accent-dark);
}

.npc-list {
  display: grid;
  gap: 8px;
}

.npc-button {
  padding: 10px;
  text-align: left;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfcfa;
  cursor: pointer;
}

.npc-button.active {
  border-color: var(--accent);
  background: var(--player);
}

.chat-panel {
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto auto;
  min-width: 0;
}

.character-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px;
  border-bottom: 1px solid var(--line);
}

.npc-name {
  font-size: 22px;
  font-weight: 800;
}

.session-badge {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 6px 8px;
  border-radius: 6px;
  background: #f0f3f1;
  color: var(--muted);
  font-size: 12px;
}

.persona {
  padding: 12px 16px;
  border-bottom: 1px solid var(--line);
}

.chat-log {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 360px;
  overflow-y: auto;
  padding: 16px;
}

.message {
  max-width: min(76ch, 86%);
  padding: 10px 12px;
  border-radius: 8px;
  line-height: 1.5;
}

.message.player {
  align-self: flex-end;
  background: var(--player);
}

.message.npc {
  align-self: flex-start;
  background: var(--npc);
}

.chat-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 96px;
  gap: 8px;
  padding: 16px;
  border-top: 1px solid var(--line);
}

.error-box {
  margin: 0 16px 16px;
  padding: 10px 12px;
  border-radius: 6px;
  background: #fff1f0;
  color: #9a2c1f;
}

.status-grid {
  display: grid;
  gap: 10px;
}

.status-item,
.fallback-box {
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfcfa;
}

.status-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.fallback-box {
  margin-top: 14px;
}

.fallback-box p {
  margin-top: 6px;
  line-height: 1.4;
}

@media (max-width: 920px) {
  .app-shell {
    grid-template-columns: 1fr;
  }

  .chat-log {
    min-height: 300px;
  }
}
```

- [ ] **Step 5: Create `web_static/app.js`**

Create `web_static/app.js`:

```javascript
const state = {
  npcs: [],
  selectedNpcId: null,
  sessionId: null,
  playerId: "demo_player",
};

const elements = {
  npcList: document.getElementById("npcList"),
  modeSelect: document.getElementById("modeSelect"),
  chatLog: document.getElementById("chatLog"),
  messageInput: document.getElementById("messageInput"),
  sendButton: document.getElementById("sendButton"),
  chatForm: document.getElementById("chatForm"),
  npcName: document.getElementById("npcName"),
  npcTitle: document.getElementById("npcTitle"),
  npcPersona: document.getElementById("npcPersona"),
  sessionBadge: document.getElementById("sessionBadge"),
  intentStatus: document.getElementById("intentStatus"),
  ragStatus: document.getElementById("ragStatus"),
  toolStatus: document.getElementById("toolStatus"),
  memoryStatus: document.getElementById("memoryStatus"),
  fallbackStatus: document.getElementById("fallbackStatus"),
  errorBox: document.getElementById("errorBox"),
};

async function loadNpcs() {
  const response = await fetch("/api/npcs");
  const body = await response.json();
  state.npcs = body.npcs;
  renderNpcList();
  if (state.npcs.length > 0) {
    await selectNpc(state.npcs[0].id);
  }
}

function renderNpcList() {
  elements.npcList.innerHTML = "";
  state.npcs.forEach((npc) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `npc-button${npc.id === state.selectedNpcId ? " active" : ""}`;
    button.textContent = `${npc.name} · ${npc.title}`;
    button.addEventListener("click", () => selectNpc(npc.id));
    elements.npcList.appendChild(button);
  });
}

async function selectNpc(npcId) {
  clearError();
  state.selectedNpcId = npcId;
  renderNpcList();

  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      npc_id: npcId,
      player_id: state.playerId,
      mode: elements.modeSelect.value,
    }),
  });
  const body = await response.json();
  if (!response.ok) {
    showError(body.detail || "Failed to create session");
    return;
  }

  state.sessionId = body.session_id;
  elements.npcName.textContent = body.npc.name;
  elements.npcTitle.textContent = body.npc.title;
  elements.npcPersona.textContent = body.npc.personality;
  elements.sessionBadge.textContent = body.session_id;
  elements.chatLog.innerHTML = "";
  appendMessage("npc", body.npc.greeting);
  resetStatus();
}

async function sendMessage(message) {
  appendMessage("player", message);
  elements.messageInput.value = "";
  elements.sendButton.disabled = true;
  clearError();

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      npc_id: state.selectedNpcId,
      player_id: state.playerId,
      message,
      mode: elements.modeSelect.value,
    }),
  });
  const body = await response.json();
  elements.sendButton.disabled = false;

  if (!response.ok) {
    showError(body.detail || "Message failed");
    return;
  }

  appendMessage("npc", body.reply);
  renderMetadata(body.metadata);
}

function appendMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  elements.chatLog.appendChild(message);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

function renderMetadata(metadata) {
  elements.intentStatus.textContent = metadata.intent || "-";
  elements.ragStatus.textContent = metadata.rag_used ? "Used" : "No";
  elements.toolStatus.textContent = metadata.tool_used ? "Used" : "No";
  elements.memoryStatus.textContent = metadata.memory_recorded ? `Turn ${metadata.turn_count}` : "Not recorded";
  elements.fallbackStatus.textContent = metadata.fallback || "None";
}

function resetStatus() {
  elements.intentStatus.textContent = "-";
  elements.ragStatus.textContent = "-";
  elements.toolStatus.textContent = "-";
  elements.memoryStatus.textContent = "-";
  elements.fallbackStatus.textContent = "None";
}

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.hidden = false;
}

function clearError() {
  elements.errorBox.textContent = "";
  elements.errorBox.hidden = true;
}

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message) {
    showError("Message cannot be empty");
    return;
  }
  await sendMessage(message);
});

elements.modeSelect.addEventListener("change", async () => {
  if (state.selectedNpcId) {
    await selectNpc(state.selectedNpcId);
  }
});

loadNpcs().catch((error) => {
  showError(error.message);
});
```

- [ ] **Step 6: Run static and API tests**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest tests/test_web_static.py tests/test_web_app.py -v
```

Expected: all static and API tests pass.

- [ ] **Step 7: Commit static UI**

Run:

```bash
git add web_static/index.html web_static/styles.css web_static/app.js tests/test_web_static.py
git commit -m "feat: add local web chat ui"
```

Expected: a commit containing only static UI and static tests.

---

### Task 5: Update README and Run Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README web demo instructions**

Add this section after the existing "Run the dialogue prototype" instructions:

~~~markdown
**5. Run the local web dialogue demo**

```bash
/Applications/anaconda3/bin/python3.13 web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The web demo starts in `Mock` mode, which does not require `ANTHROPIC_API_KEY`. Mock mode uses the existing NPC config, intent router, and memory logging, but the response text is deterministic local fallback text rather than Claude output.

Switch to `Live` mode only after configuring:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

If the API key is missing, the web demo falls back to mock mode and shows the fallback reason in the status panel.
~~~

- [ ] **Step 2: Run full test suite**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest
```

Expected:

```text
all tests pass
```

The exact count will increase from the previous 64 tests after adding web tests.

- [ ] **Step 3: Run local server smoke check**

Run:

```bash
/Applications/anaconda3/bin/python3.13 web_app.py
```

Expected terminal output includes Uvicorn startup on `http://127.0.0.1:8000`.

Open `http://127.0.0.1:8000` and manually verify:

- NPC list appears.
- First NPC greeting appears.
- Sending `我背包里有没有月盐药剂？` returns an NPC mock reply.
- Status panel shows `inventory_query`, `Tool: Used`, and memory turn count.
- Switching to `Live` without `ANTHROPIC_API_KEY` returns a visible fallback reason instead of crashing.

- [ ] **Step 4: Stop local server**

Press `Ctrl+C` in the server terminal.

Expected: server exits cleanly and no long-running process remains.

- [ ] **Step 5: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: document local web demo"
```

Expected: a commit containing only README documentation.

---

### Task 6: Final Integration Check and Push

**Files:**
- Verify: all changed files

- [ ] **Step 1: Check working tree**

Run:

```bash
git status --short --branch
```

Expected: clean working tree on `codex/phase4-memory-compact`.

- [ ] **Step 2: Run full verification again**

Run:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest
```

Expected: all tests pass.

- [ ] **Step 3: Check recent commits**

Run:

```bash
git log --oneline --decorate --max-count=8
```

Expected: recent commits include dependency, service, API, UI, and README changes.

- [ ] **Step 4: Push branch and default branch**

Run:

```bash
git push origin codex/phase4-memory-compact
git push origin codex/phase4-memory-compact:主
```

Expected: GitHub branch and default branch both update without force push.

- [ ] **Step 5: Final response**

Report:

- local URL
- files changed
- tests run
- manual smoke-check result
- current GitHub branch/default branch status
- whether live Claude mode was actually tested or only guarded/fallback-tested
