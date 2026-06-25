# Local Web UI Design

## Purpose

Build a simple local web chat interface for GameNPCDialogue that can run reliably on the user's machine and present the project as an AI agent orchestration demo.

The UI should support two execution modes:

- `mock`: fully local fallback replies with no API key required.
- `live`: use the existing Claude/RAG/tool/memory pipeline when `ANTHROPIC_API_KEY` and local dependencies are available.

The first implementation should be mock-first so the demo always works locally. The live pipeline should be connected after the web API boundary is stable.

## Current Evidence

Observed project structure:

- `npc_dialogue.py` contains the current CLI loop, NPC loading, routing, RAG trigger logic, Claude calls, tool-calling path, quest generation, GM feedback, and memory recording.
- `npc_config.json` contains four NPC definitions: `blacksmith`, `innkeeper`, `mysterious_wizard`, and `gm_advisor`.
- `intent_router.py` exposes deterministic intent classification.
- `game_tools.py`, `claude_tools.py`, and `tool_executor.py` provide simulated backend APIs, Anthropic tool schemas, and local tool execution.
- `memory_store.py` provides session logging, recent conversation history, compact trigger state, story events, and quest state.
- `rag_pipeline.py` owns local Markdown knowledge retrieval with Chroma and HuggingFace embeddings.
- `tests/` contains offline tests for routing, RAG, prompts, tools, memory, and dialogue behavior.

The current interface is terminal-first. The new UI should reuse these modules instead of duplicating business logic in frontend code.

## Chosen Direction

Use a small FastAPI backend with static HTML/CSS/JS frontend.

Rejected alternatives:

- Streamlit or Gradio: faster, but weaker for portfolio presentation and API-boundary explanation.
- React plus FastAPI: polished, but heavier than needed for a local demo and adds Node/Vite complexity.

The chosen design is intentionally conservative: one Python backend process serves both APIs and static files, while the frontend remains plain HTML/CSS/JS.

## User Experience

The user opens a local URL and sees:

- NPC selector with the four existing characters.
- Character panel with name, title, greeting, and persona summary.
- Chat transcript with player and NPC message bubbles.
- Message input and send button.
- Mode selector: `Mock` or `Live`.
- Status panel showing intent, RAG status, tool status, memory/session status, and fallback warnings.

The interface should feel like a compact game dialogue console, not a marketing landing page. It should prioritize readability, fast repeated interaction, and clear system-state visibility.

## Architecture

```text
Browser UI
  -> FastAPI static files
  -> FastAPI JSON endpoints
  -> dialogue_service.py
  -> existing project modules
```

Planned files:

```text
web_app.py
dialogue_service.py
web_static/
  index.html
  styles.css
  app.js
```

`web_app.py` owns HTTP endpoints and static serving.

`dialogue_service.py` owns reusable dialogue orchestration for web mode. It should provide a small API that can later be reused by CLI code. The initial service can implement mock mode first and add live mode in a scoped follow-up step.

`web_static/` contains the browser UI. The frontend should not know how RAG, tool calling, or memory works internally; it should only display metadata returned by the backend.

## API Design

### `GET /api/npcs`

Returns available NPCs from `npc_config.json`.

Response shape:

```json
{
  "npcs": [
    {
      "id": "blacksmith",
      "name": "格林大叔",
      "title": "镇上的老铁匠",
      "greeting": "..."
    }
  ]
}
```

### `POST /api/sessions`

Creates a session for one NPC and player.

Request shape:

```json
{
  "npc_id": "blacksmith",
  "player_id": "demo_player",
  "mode": "mock"
}
```

Response shape:

```json
{
  "session_id": "blacksmith_demo_player_...",
  "npc": {
    "id": "blacksmith",
    "name": "格林大叔",
    "title": "镇上的老铁匠",
    "greeting": "..."
  },
  "mode": "mock"
}
```

### `POST /api/chat`

Sends one player message and returns one NPC reply plus diagnostic metadata.

Request shape:

```json
{
  "session_id": "blacksmith_demo_player_...",
  "npc_id": "blacksmith",
  "player_id": "demo_player",
  "message": "我现在应该做什么？",
  "mode": "mock"
}
```

Response shape:

```json
{
  "reply": "...",
  "metadata": {
    "intent": "inventory_query",
    "rag_used": false,
    "tool_used": true,
    "memory_recorded": true,
    "mode": "mock",
    "fallback": null
  }
}
```

## Mock Mode

Mock mode should:

- Require no API key.
- Use `intent_router.classify_intent()` for visible intent metadata.
- Use existing NPC config for character name, title, greeting, and rough style.
- Return deterministic local replies for common intents: persona chat, lore question, quest request, inventory query, and GM feedback.
- Record memory through `DialogueMemorySession` when feasible, so the UI can demonstrate session logging without a live model call.

Mock mode must clearly label itself as mock output. It should not pretend to be Claude-generated.

## Live Mode

Live mode should:

- Require `ANTHROPIC_API_KEY`.
- Reuse the existing Claude, RAG, tool-calling, quest, GM feedback, and memory code paths as much as possible.
- Return fallback metadata if the live call fails.
- Avoid crashing the web server when model calls, RAG initialization, Chroma, or embedding downloads fail.

The first implementation may expose live mode as a guarded path. If it is not fully connected yet, the UI should say so explicitly and fall back to mock mode.

## Error Handling

The backend should return structured errors for:

- unknown NPC ID
- empty message
- unknown session
- unavailable live mode
- internal exception during live call

The frontend should show errors inline near the chat input and keep the current transcript visible.

## Testing

Backend tests should cover:

- `GET /api/npcs`
- `POST /api/sessions`
- mock `POST /api/chat`
- invalid NPC ID
- empty message
- live mode without `ANTHROPIC_API_KEY` falls back or returns a clear unavailable response

Existing tests should continue to pass:

```bash
/Applications/anaconda3/bin/python3.13 -m pytest
```

After implementation, run a local smoke test:

```bash
/Applications/anaconda3/bin/python3.13 web_app.py
```

Then open the local URL and send at least one message to each NPC in mock mode.

## Success Criteria

- The app runs locally from one command.
- The user can select an NPC and chat through a browser.
- Mock mode works without external API credentials.
- The UI displays intent, RAG, tool, and memory status.
- Existing tests still pass.
- New web API tests pass.
- README explains how to run the web demo.

## Scope Boundaries

In scope:

- Local web UI.
- FastAPI backend.
- Mock mode.
- Guarded live mode path.
- Basic tests.
- README update.

Out of scope for the first implementation:

- User authentication.
- Deployment.
- React/Vite.
- Streaming token output.
- Persistent browser accounts.
- Full game engine integration.
- Image generation for NPC portraits.

