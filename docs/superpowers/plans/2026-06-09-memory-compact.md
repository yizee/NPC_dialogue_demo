# Memory Compact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent memory foundation that stores full NPC dialogue logs while keeping Claude live prompts focused with recent history and compact summaries.

**Architecture:** Add a focused `memory_store.py` module for session state, JSONL persistence, recent-window trimming, compact trigger checks, story events, and quest state. Add compact prompt construction to `prompts.py`, then update `npc_dialogue.py` to use memory state instead of passing full `conversation_history` into every live Claude call.

**Tech Stack:** Python standard library (`dataclasses`, `json`, `datetime`, `pathlib`), Anthropic Messages API integration already present in `npc_dialogue.py`, pytest offline unit tests.

---

## File Structure

- Create `memory_store.py`
  - Owns memory data structures and persistence.
  - Provides JSON-serializable session state.
  - Does not import Anthropic, LangChain, or NPC prompt code.
- Create `tests/test_memory_store.py`
  - Tests session creation, JSONL append, recent-window trimming, compact trigger checks, story events, quest state, and compact input construction.
- Modify `prompts.py`
  - Add `build_compact_prompt()`.
  - Add optional `memory_summary` and `quest_state` support to `build_system_prompt()`.
- Modify `tests/test_prompts.py`
  - Test compact prompt contract and system prompt memory injection.
- Modify `npc_dialogue.py`
  - Replace local `conversation_history` ownership with `DialogueMemorySession`.
  - Pass recent history to live persona and quest workflows.
  - Preserve full in-memory conversation history for `quit` GM feedback within the current session.
  - Append every successful exchange to raw JSONL.
  - Keep Phase 1 RAG, Phase 2 router, and Phase 3 tool calling behavior unchanged.
- Modify `tests/test_dialogue_routing.py`
  - Test new helper behavior without calling Claude.
- Modify `README.md`
  - Document Phase 4 memory behavior and log location.

---

## Task 1: Memory Store Core

**Files:**
- Create: `memory_store.py`
- Test: `tests/test_memory_store.py`

- [ ] **Step 1: Write failing tests for session creation and raw JSONL append**

Add this to `tests/test_memory_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_memory_store.py -q
```

Expected: FAIL because `memory_store.py` does not exist.

- [ ] **Step 3: Implement session id and raw append**

Create `memory_store.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re


MAX_RECENT_MESSAGES = 12
COMPACT_EVERY_N_TURNS = 8


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def create_session_id(npc_id: str, player_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_safe_id(npc_id)}_{_safe_id(player_id)}"


@dataclass
class DialogueMemorySession:
    npc_id: str
    player_id: str
    log_dir: str | Path = "logs/sessions"
    max_recent_messages: int = MAX_RECENT_MESSAGES
    compact_every_n_turns: int = COMPACT_EVERY_N_TURNS
    session_id: str | None = None
    recent_conversation_history: list[dict] = field(default_factory=list)
    session_summary: str = ""
    full_conversation_history: list[dict] = field(default_factory=list)
    story_events: list[dict] = field(default_factory=list)
    quest_state: dict = field(default_factory=lambda: {
        "active_quest": None,
        "completed_quests": [],
        "failed_quests": [],
        "known_clues": [],
        "pending_commitments": [],
    })
    turn_count: int = 0
    compact_pending: bool = False

    def __post_init__(self) -> None:
        if self.session_id is None:
            self.session_id = create_session_id(self.npc_id, self.player_id)
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_log_path = self.log_dir / f"{self.session_id}.jsonl"

    def append_turn(
        self,
        user_input: str,
        assistant_reply: str,
        intent: str,
        used_rag: bool = False,
        used_tool_calling: bool = False,
        rag_sources: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> dict:
        self.turn_count += 1
        raw_turn = {
            "session_id": self.session_id,
            "turn_index": self.turn_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "npc_id": self.npc_id,
            "player_id": self.player_id,
            "user_input": user_input,
            "assistant_reply": assistant_reply,
            "intent": intent,
            "used_rag": used_rag,
            "used_tool_calling": used_tool_calling,
            "rag_sources": rag_sources or [],
            "tool_names": tool_names or [],
        }
        with self.raw_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_turn, ensure_ascii=False) + "\n")
        self.recent_conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ])
        self.full_conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ])
        if len(self.recent_conversation_history) > self.max_recent_messages:
            self.compact_pending = True
        self.trim_recent_history()
        return raw_turn

    def trim_recent_history(self) -> None:
        if len(self.recent_conversation_history) > self.max_recent_messages:
            self.recent_conversation_history = self.recent_conversation_history[-self.max_recent_messages:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_memory_store.py -q
```

Expected: PASS for the two tests in this task.

- [ ] **Step 5: Commit**

Only commit after confirming the broader worktree state with the user, because Phase 3 currently has uncommitted files. If committing this task alone is approved:

```bash
git add memory_store.py tests/test_memory_store.py
git commit -m "feat: add dialogue memory store"
```

---

## Task 2: Recent Window, Compact Trigger, Story Events, and Quest State

**Files:**
- Modify: `memory_store.py`
- Modify: `tests/test_memory_store.py`

- [ ] **Step 1: Write failing tests for memory behavior**

Append to `tests/test_memory_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_memory_store.py -q
```

Expected: FAIL because `should_compact()`, `add_story_event()`, and `update_quest_state()` do not exist.

- [ ] **Step 3: Implement memory behavior**

Add these methods to `DialogueMemorySession` in `memory_store.py`:

```python
    def should_compact(self, force: bool = False) -> bool:
        if force:
            return True
        if self.compact_pending:
            return True
        if len(self.recent_conversation_history) > self.max_recent_messages:
            return True
        return self.turn_count > 0 and self.turn_count % self.compact_every_n_turns == 0

    def add_story_event(self, event_type: str, summary: str) -> dict:
        event = {
            "event_type": event_type,
            "summary": summary,
        }
        self.story_events.append(event)
        return event

    def update_quest_state(self, **updates) -> dict:
        for key, value in updates.items():
            if key not in self.quest_state:
                raise KeyError(f"Unknown quest_state key: {key}")
            self.quest_state[key] = value
        return self.quest_state

    def mark_compacted(self) -> None:
        self.compact_pending = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_memory_store.py -q
```

Expected: PASS.

---

## Task 3: Compact Input and Prompt Contract

**Files:**
- Modify: `memory_store.py`
- Modify: `prompts.py`
- Modify: `tests/test_memory_store.py`
- Modify: `tests/test_prompts.py`

- [ ] **Step 1: Write failing compact input test**

Append to `tests/test_memory_store.py`:

```python
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
```

- [ ] **Step 2: Write failing compact prompt tests**

Append to `tests/test_prompts.py`:

```python
from prompts import build_compact_prompt


def test_build_compact_prompt_defines_structured_output_contract():
    npc = {
        "name": "神秘法师",
        "title": "星尘塔的守门人",
        "topics": ["魔法", "旧塔"],
    }
    player_context = {
        "name": "旅行者",
        "level": 5,
        "quest_status": "尚未接取任务",
        "reputation": "中立",
    }
    compact_input = {
        "session_summary": "玩家想学习魔法。",
        "recent_conversation_history": [
            {"role": "user", "content": "我想学习魔法。"},
            {"role": "assistant", "content": "先证明你的耐心。"},
        ],
        "story_events": [{"event_type": "clue_revealed", "summary": "旧塔有魔法书。"}],
        "quest_state": {"active_quest": "wizard_starter_1"},
    }

    prompt = build_compact_prompt(npc, player_context, compact_input)

    assert "session_summary" in prompt
    assert "important_facts" in prompt
    assert "quest_progress" in prompt
    assert "player_commitments" in prompt
    assert "npc_attitude" in prompt
    assert "story_events" in prompt
    assert "open_threads" in prompt
    assert "不要判断任务完成" in prompt
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_memory_store.py tests/test_prompts.py -q
```

Expected: FAIL because `build_compact_input()` and `build_compact_prompt()` do not exist.

- [ ] **Step 4: Implement compact input**

Add this method to `DialogueMemorySession`:

```python
    def build_compact_input(self) -> dict:
        return {
            "session_id": self.session_id,
            "npc_id": self.npc_id,
            "player_id": self.player_id,
            "session_summary": self.session_summary,
            "full_conversation_history": list(self.full_conversation_history),
            "recent_conversation_history": list(self.recent_conversation_history),
            "story_events": list(self.story_events),
            "quest_state": dict(self.quest_state),
        }
```

- [ ] **Step 5: Implement compact prompt**

Add to `prompts.py`:

```python
def build_compact_prompt(npc: dict, player_context: dict, compact_input: dict) -> str:
    recent_history = "\n".join([
        f"{'玩家' if m['role'] == 'user' else npc['name']}: {m['content']}"
        for m in compact_input.get("recent_conversation_history", [])
    ]) or "（无近期对话）"

    return f"""你是一个 NPC 对话记忆整理器。你的任务是把长对话压缩成结构化记忆，供后续 NPC 对话使用。

== NPC 信息 ==
姓名：{npc['name']}
身份：{npc['title']}
擅长话题：{', '.join(npc['topics'])}

== 玩家信息 ==
玩家名称：{player_context['name']}
玩家等级：{player_context['level']}
当前状态：{player_context['quest_status']}
声望：{player_context['reputation']}

== 已有摘要 ==
{compact_input.get('session_summary') or '（暂无摘要）'}

== 近期对话 ==
{recent_history}

== 已记录剧情事件 ==
{compact_input.get('story_events', [])}

== 当前任务状态 ==
{compact_input.get('quest_state', {})}

请只输出 JSON 对象，字段必须包含：
- session_summary
- important_facts
- quest_progress
- player_commitments
- npc_attitude
- story_events
- open_threads

保留任务进度、NPC 态度、玩家承诺、重要线索和可能影响后续剧情的世界状态。
删除重复寒暄、无意义闲聊、重复表达和不影响任务或剧情的细节。
不要判断任务完成，不要解锁结局，不要把猜测写成事实。
"""
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_memory_store.py tests/test_prompts.py -q
```

Expected: PASS.

---

## Task 4: System Prompt Memory Injection

**Files:**
- Modify: `prompts.py`
- Modify: `tests/test_prompts.py`

- [ ] **Step 1: Write failing system prompt memory test**

Append to `tests/test_prompts.py`:

```python
from prompts import build_system_prompt


def test_build_system_prompt_includes_memory_summary_and_quest_state():
    npc = {
        "name": "神秘法师",
        "title": "星尘塔的守门人",
        "personality": "谨慎",
        "speech_style": "含蓄",
        "background": "守护旧塔秘密。",
        "topics": ["魔法"],
        "forbidden_topics": ["真实身份"],
    }
    player_context = {
        "name": "旅行者",
        "level": 5,
        "quest_status": "尚未接取任务",
        "reputation": "中立",
    }
    quest_state = {
        "active_quest": "wizard_starter_1",
        "known_clues": ["旧塔可能藏有入门魔法书"],
    }

    prompt = build_system_prompt(
        npc,
        player_context,
        memory_summary="玩家想学习魔法，并已获得旧塔线索。",
        quest_state=quest_state,
    )

    assert "== 对话记忆 ==" in prompt
    assert "玩家想学习魔法" in prompt
    assert "== 任务状态 ==" in prompt
    assert "wizard_starter_1" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_prompts.py::test_build_system_prompt_includes_memory_summary_and_quest_state -q
```

Expected: FAIL because `build_system_prompt()` does not accept `memory_summary` or `quest_state`.

- [ ] **Step 3: Update system prompt signature and sections**

Modify `build_system_prompt()` in `prompts.py`:

```python
def build_system_prompt(
    npc: dict,
    player_context: dict,
    retrieved_context: str = "",
    memory_summary: str = "",
    quest_state: dict | None = None,
) -> str:
```

Add before `return`:

```python
    memory_section = ""
    if memory_summary:
        memory_section = f"""

== 对话记忆 ==
{memory_summary}
"""

    quest_state_section = ""
    if quest_state:
        quest_state_section = f"""

== 任务状态 ==
{quest_state}
"""
```

Insert `{memory_section}` and `{quest_state_section}` after the player information block in the returned prompt.

- [ ] **Step 4: Run prompt tests**

Run:

```bash
python3 -m pytest tests/test_prompts.py -q
```

Expected: PASS.

---

## Task 5: NPC Dialogue Integration Helpers

**Files:**
- Modify: `npc_dialogue.py`
- Modify: `tests/test_dialogue_routing.py`

- [ ] **Step 1: Write failing helper tests**

Append to `tests/test_dialogue_routing.py`:

```python
from memory_store import DialogueMemorySession
from npc_dialogue import record_memory_turn


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py::test_record_memory_turn_appends_exchange -q
```

Expected: FAIL because `record_memory_turn()` does not exist.

- [ ] **Step 3: Add memory helper**

Add imports to `npc_dialogue.py`:

```python
from memory_store import DialogueMemorySession
```

Add helper near the routing helpers:

```python
def record_memory_turn(
    memory_session: DialogueMemorySession,
    user_input: str,
    assistant_reply: str,
    intent: str,
    used_rag: bool = False,
    used_tool_calling: bool = False,
    rag_sources: list[str] | None = None,
    tool_names: list[str] | None = None,
) -> None:
    try:
        memory_session.append_turn(
            user_input=user_input,
            assistant_reply=assistant_reply,
            intent=intent,
            used_rag=used_rag,
            used_tool_calling=used_tool_calling,
            rag_sources=rag_sources or [],
            tool_names=tool_names or [],
        )
    except Exception as exc:
        print(f"[Memory 警告] 对话日志保存失败，将继续本轮对话：{exc}\n")
```

- [ ] **Step 4: Run helper test**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py::test_record_memory_turn_appends_exchange -q
```

Expected: PASS.

---

## Task 6: Replace Full Conversation History in Live Chat

**Files:**
- Modify: `npc_dialogue.py`
- Modify: `tests/test_dialogue_routing.py`

- [ ] **Step 1: Write failing test for live message source**

Append to `tests/test_dialogue_routing.py`:

```python
from npc_dialogue import get_live_conversation_history


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py::test_get_live_conversation_history_uses_recent_memory_only -q
```

Expected: FAIL because `get_live_conversation_history()` does not exist.

- [ ] **Step 3: Add live history helper**

Add to `npc_dialogue.py`:

```python
def get_live_conversation_history(memory_session: DialogueMemorySession) -> list[dict]:
    return list(memory_session.recent_conversation_history)
```

- [ ] **Step 4: Integrate memory session into `chat_with_npc()`**

Inside `chat_with_npc()` replace:

```python
    conversation_history = []
```

with:

```python
    player_id = player_context.get("player_id", "player_001")
    memory_session = DialogueMemorySession(npc_id=npc_id, player_id=player_id)
```

Update existing uses:

```python
conversation_history
```

to:

```python
get_live_conversation_history(memory_session)
```

where history is passed into Claude for live persona, quest, or GM prompts.

Exception: for `quit` GM feedback, use the full current-session history:

```python
memory_session.full_conversation_history
```

This preserves the old GM behavior while still preventing normal live dialogue from sending the full raw history every turn.

After each generated response, replace direct `conversation_history.append(...)` pairs with:

```python
record_memory_turn(
    memory_session=memory_session,
    user_input=player_input,
    assistant_reply=npc_reply,
    intent=route.intent,
    used_rag=bool(retrieved_context),
    used_tool_calling=False,
    rag_sources=[],
    tool_names=[],
)
```

For `/quest`, use:

```python
record_memory_turn(
    memory_session=memory_session,
    user_input="（玩家请求一个任务）",
    assistant_reply=quest_text,
    intent="quest_request",
)
```

For tool calling, use:

```python
record_memory_turn(
    memory_session=memory_session,
    user_input=player_input,
    assistant_reply=tool_response,
    intent=route.intent,
    used_tool_calling=True,
    tool_names=["check_inventory"],
)
```

- [ ] **Step 5: Update system prompt call**

In the normal persona/RAG path, call:

```python
        system_prompt = build_system_prompt(
            npc,
            player_context,
            retrieved_context=retrieved_context,
            memory_summary=memory_session.session_summary,
            quest_state=memory_session.quest_state,
        )
```

Then send:

```python
        live_history = get_live_conversation_history(memory_session)
        live_history.append({"role": "user", "content": player_input})
```

and pass `messages=live_history`.

- [ ] **Step 6: Run dialogue tests**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py tests/test_dialogue_tool_calling.py -q
```

Expected: PASS.

---

## Task 7: Compact Orchestration Stub

**Files:**
- Modify: `npc_dialogue.py`
- Modify: `tests/test_dialogue_routing.py`

- [ ] **Step 1: Write failing compact orchestration test**

Append to `tests/test_dialogue_routing.py`:

```python
from npc_dialogue import compact_memory_if_needed


def test_compact_memory_if_needed_keeps_existing_summary_without_api_call(tmp_path):
    session = DialogueMemorySession(
        npc_id="blacksmith",
        player_id="player_001",
        log_dir=tmp_path,
    )
    session.session_summary = "旧摘要"

    compact_memory_if_needed(session, force=True)

    assert session.session_summary == "旧摘要"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py::test_compact_memory_if_needed_keeps_existing_summary_without_api_call -q
```

Expected: FAIL because `compact_memory_if_needed()` does not exist.

- [ ] **Step 3: Add compact stub**

Add to `npc_dialogue.py`:

```python
def compact_memory_if_needed(memory_session: DialogueMemorySession, force: bool = False) -> None:
    if not memory_session.should_compact(force=force):
        return
    # Phase 4 MVP defines the compact contract and trigger point.
    # Claude-based compact generation will be added only after storage and tests are stable.
    memory_session.trim_recent_history()
    memory_session.mark_compacted()
```

Call it after `record_memory_turn(...)` in each successful workflow:

```python
compact_memory_if_needed(memory_session)
```

Call it on `quit` before GM feedback:

```python
compact_memory_if_needed(memory_session, force=True)
```

- [ ] **Step 4: Run dialogue tests**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py -q
```

Expected: PASS.

---

## Task 8: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Memory Compact section**

Add a section near the architecture overview:

~~~markdown
## Memory and Compact Layer

The project keeps full raw conversation logs while passing only recent and compressed context into live Claude calls.

```text
each valid turn
  -> append raw JSONL log under logs/sessions/
  -> update recent_conversation_history
  -> update story_events / quest_state when applicable
  -> compact trigger point checks whether summary refresh is needed
```

Compact is not deletion. Raw logs remain available for audit, debugging, replay, GM feedback, and future final-story generation. Compact summaries are used only to keep live NPC dialogue focused and token-efficient.

Quest state is stored separately from compact text. A summary may say that the player promised to find a Dragon Hunter weapon, but quest completion must still be validated by explicit game state or tool/API logic.
~~~

- [ ] **Step 2: Add file tree entries**

Update the project tree section:

```text
├── memory_store.py          # Persistent raw logs, recent history, compact triggers, story and quest state
```

and:

```text
└── logs/sessions/*.jsonl    # Full raw dialogue session logs
```

- [ ] **Step 3: No test command required**

README-only change has no direct unit test. Verify formatting by reading the edited section:

```bash
rg -n "Memory and Compact|memory_store.py|logs/sessions" README.md
```

Expected: all three entries are found.

---

## Task 9: Full Verification

**Files:**
- No new edits unless verification exposes a bug.

- [ ] **Step 1: Run the full offline test suite**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run a CLI quit smoke test**

Run:

```bash
printf "blacksmith\nquit\n" | python3 npc_dialogue.py
```

Expected:
- Program starts.
- Blacksmith NPC loads.
- `quit` exits cleanly.
- If there is no conversation history, GM feedback generation is skipped or does not crash.

- [ ] **Step 3: Verify logs directory behavior after a short session**

Run:

```bash
printf "blacksmith\n你好\nquit\n" | python3 npc_dialogue.py
```

Expected:
- The normal chat may call Claude and requires `ANTHROPIC_API_KEY`.
- If run with a valid key, a new `logs/sessions/*.jsonl` file is created.
- The JSONL file contains the user input and NPC reply.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected:
- Phase 4 files appear as modified or untracked.
- Existing Phase 3 uncommitted files remain untouched unless explicitly staged later.

---

## Self-Review Against Spec

- Raw JSONL persistence: Task 1 and Task 6.
- Recent conversation window: Task 1, Task 2, Task 6.
- Compact summary structure: Task 3.
- Story events: Task 2.
- Quest state separate from compact: Task 2, Task 4, Task 8.
- Offline tests: Tasks 1-7 and Task 9.
- Preserve RAG/router/tool calling behavior: Task 6 and Task 9.
- No final novel generation: deliberately excluded from all tasks.
- No real database or UI: deliberately excluded from all tasks.
