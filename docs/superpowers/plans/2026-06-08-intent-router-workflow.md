# Intent Router Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic intent router and local tool/API workflow so ordinary NPC inputs route to persona chat, RAG lore answers, quest generation, inventory tools, or GM feedback before calling Claude.

**Architecture:** `intent_router.py` classifies input with deterministic Chinese/English keyword rules and returns a small structured result. `game_tools.py` owns local player-profile and inventory-style tool functions. `npc_dialogue.py` keeps `quit` and `/quest` as explicit commands, then routes normal input through the router; only `lore_question` initializes RAG, while persona chat avoids RAG.

**Tech Stack:** Python 3.13, pytest, existing Anthropic SDK integration, existing LangChain/Chroma RAG service

---

## File Map

- Create `intent_router.py`: intent labels, `RouteResult`, deterministic `classify_intent()`.
- Create `game_tools.py`: local API-style player profile, inventory check, next-action recommendation, deterministic response formatters.
- Create `tests/test_intent_router.py`: offline router classification tests.
- Create `tests/test_game_tools.py`: offline local tool behavior tests.
- Modify `npc_dialogue.py`: route normal player input through the router and local tools.
- Modify `README.md`: document Phase 2 router and tool/API workflow.

> Repository note: `/Users/shiyize/Desktop/GameNPCDialogue` is inside a Git repository rooted at `/Users/shiyize`. Do not stage or commit unless repository isolation is fixed or the user explicitly approves staging through the parent repository.

### Task 1: Implement Deterministic Intent Router

**Files:**
- Create: `intent_router.py`
- Create: `tests/test_intent_router.py`

- [ ] **Step 1: Write failing router tests**

Create `tests/test_intent_router.py`:

```python
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
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_intent_router.py -v
```

Expected: collection failure with `ModuleNotFoundError: No module named 'intent_router'`.

- [ ] **Step 3: Implement router**

Create `intent_router.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


PERSONA_CHAT = "persona_chat"
LORE_QUESTION = "lore_question"
QUEST_REQUEST = "quest_request"
INVENTORY_QUERY = "inventory_query"
GM_FEEDBACK = "gm_feedback"

VALID_INTENTS = {
    PERSONA_CHAT,
    LORE_QUESTION,
    QUEST_REQUEST,
    INVENTORY_QUERY,
    GM_FEEDBACK,
}


@dataclass(frozen=True)
class RouteResult:
    intent: str
    confidence: float
    reason: str


KEYWORDS_BY_INTENT = {
    GM_FEEDBACK: [
        "投诉",
        "反馈",
        "举报",
        "卡住",
        "不合理",
        "bug",
        "complaint",
        "feedback",
        "report",
        "stuck",
        "broken",
    ],
    INVENTORY_QUERY: [
        "背包",
        "物品",
        "道具",
        "有没有",
        "是否拥有",
        "我有",
        "inventory",
        "backpack",
        "item",
        "do i have",
        "have the",
    ],
    QUEST_REQUEST: [
        "接任务",
        "任务",
        "委托",
        "有什么事要我做",
        "给我一个任务",
        "quest",
        "mission",
        "task",
        "something to do",
    ],
    LORE_QUESTION: [
        "是什么",
        "谁是",
        "在哪里",
        "为什么",
        "世界观",
        "设定",
        "背景",
        "魔龙",
        "地精之王",
        "碎星髓",
        "真结局",
        "who is",
        "what is",
        "where is",
        "why",
        "lore",
        "world",
        "monster",
        "goblin king",
        "dragon",
    ],
}


def _matched_keyword(text: str, keywords: list[str]) -> str | None:
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() in lowered:
            return keyword
    return None


def classify_intent(player_input: str) -> RouteResult:
    text = player_input.strip()
    if not text:
        return RouteResult(PERSONA_CHAT, 0.0, "empty input")

    priority = [GM_FEEDBACK, INVENTORY_QUERY, QUEST_REQUEST, LORE_QUESTION]
    for intent in priority:
        matched = _matched_keyword(text, KEYWORDS_BY_INTENT[intent])
        if matched:
            return RouteResult(intent, 0.9, f"matched keyword: {matched}")

    return RouteResult(PERSONA_CHAT, 0.5, "no workflow keyword matched")
```

- [ ] **Step 4: Run router tests**

Run:

```bash
python3 -m pytest tests/test_intent_router.py -v
```

Expected: all router tests pass.

### Task 2: Implement Local Tool/API Functions

**Files:**
- Create: `game_tools.py`
- Create: `tests/test_game_tools.py`

- [ ] **Step 1: Write failing tool tests**

Create `tests/test_game_tools.py`:

```python
from game_tools import (
    check_inventory,
    format_inventory_response,
    get_player_profile,
    recommend_next_action,
)


def test_get_player_profile_returns_default_profile():
    profile = get_player_profile("player_001")

    assert profile["name"] == "旅行者"
    assert profile["level"] == 5
    assert profile["class"] == "未选择"
    assert "月盐药剂" in profile["inventory"]


def test_check_inventory_finds_known_item():
    profile = get_player_profile("player_001")

    result = check_inventory(profile, "月盐药剂")

    assert result["has_item"] is True
    assert result["item_name"] == "月盐药剂"


def test_check_inventory_rejects_missing_item():
    profile = get_player_profile("player_001")

    result = check_inventory(profile, "魔龙之心")

    assert result["has_item"] is False
    assert result["item_name"] == "魔龙之心"


def test_format_inventory_response_is_deterministic():
    profile = get_player_profile("player_001")
    found = check_inventory(profile, "月盐药剂")
    missing = check_inventory(profile, "魔龙之心")

    assert "你有月盐药剂" in format_inventory_response(found)
    assert "你还没有魔龙之心" in format_inventory_response(missing)


def test_recommend_next_action_for_unselected_class():
    profile = get_player_profile("player_001")

    recommendation = recommend_next_action(profile)

    assert "选择一个主要职业" in recommendation
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_game_tools.py -v
```

Expected: collection failure with `ModuleNotFoundError: No module named 'game_tools'`.

- [ ] **Step 3: Implement local tools**

Create `game_tools.py`:

```python
from __future__ import annotations


DEFAULT_PLAYER_PROFILE = {
    "player_id": "player_001",
    "name": "旅行者",
    "level": 5,
    "class": "未选择",
    "inventory": ["黑铁夹具", "月盐药剂"],
    "active_quest": None,
}


def get_player_profile(player_id: str) -> dict:
    profile = dict(DEFAULT_PLAYER_PROFILE)
    profile["player_id"] = player_id
    profile["inventory"] = list(DEFAULT_PLAYER_PROFILE["inventory"])
    return profile


def check_inventory(player_context: dict, item_name: str) -> dict:
    inventory = player_context.get("inventory", [])
    normalized_query = item_name.strip()
    has_item = any(normalized_query in item or item in normalized_query for item in inventory)
    return {
        "item_name": normalized_query,
        "has_item": has_item,
        "inventory": list(inventory),
    }


def format_inventory_response(inventory_result: dict) -> str:
    item_name = inventory_result["item_name"]
    if inventory_result["has_item"]:
        return f"[工具调用: check_inventory] 你有{item_name}。"
    return f"[工具调用: check_inventory] 你还没有{item_name}。"


def recommend_next_action(player_state: dict) -> str:
    if player_state.get("active_quest"):
        return f"建议继续推进当前任务：{player_state['active_quest']}。"
    if player_state.get("class", "未选择") == "未选择":
        return "建议先选择一个主要职业：可以找艾尔文走法师路线，找格林大叔走战士路线，或找罗莎老板娘走游侠路线。"
    if player_state.get("level", 1) < 5:
        return "建议先完成外围调查或采集任务提升等级。"
    return "建议调查魔龙猎人遗物，为挑战灰烬魔龙做准备。"
```

- [ ] **Step 4: Run tool tests**

Run:

```bash
python3 -m pytest tests/test_game_tools.py -v
```

Expected: all tool tests pass.

### Task 3: Add Routing Helper Around Existing Chat Behavior

**Files:**
- Modify: `npc_dialogue.py`
- Create: `tests/test_dialogue_routing.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_dialogue_routing.py`:

```python
from npc_dialogue import should_use_rag_for_intent, extract_inventory_item_name


def test_should_use_rag_only_for_lore_question():
    assert should_use_rag_for_intent("lore_question") is True
    assert should_use_rag_for_intent("persona_chat") is False
    assert should_use_rag_for_intent("inventory_query") is False
    assert should_use_rag_for_intent("quest_request") is False
    assert should_use_rag_for_intent("gm_feedback") is False


def test_extract_inventory_item_name_removes_common_question_words():
    assert extract_inventory_item_name("我背包里有月盐药剂吗？") == "月盐药剂"
    assert extract_inventory_item_name("Do I have the black iron tongs in my inventory?") == "black iron tongs"
```

- [ ] **Step 2: Run helper tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py -v
```

Expected: import failure for missing helper functions.

- [ ] **Step 3: Add imports and helper functions**

Modify imports at the top of `npc_dialogue.py`:

```python
import json
import os
import re

from anthropic import Anthropic

from game_tools import (
    check_inventory,
    format_inventory_response,
    get_player_profile,
    recommend_next_action,
)
from intent_router import (
    GM_FEEDBACK,
    INVENTORY_QUERY,
    LORE_QUESTION,
    PERSONA_CHAT,
    QUEST_REQUEST,
    classify_intent,
)
from prompts import build_gm_prompt, build_quest_prompt, build_system_prompt
from rag_pipeline import RagService
```

Add helper functions after `safety_check()`:

```python
def should_use_rag_for_intent(intent: str) -> bool:
    return intent == LORE_QUESTION


def extract_inventory_item_name(player_input: str) -> str:
    text = player_input.strip()
    replacements = [
        "我背包里有",
        "背包里有",
        "我有没有",
        "有没有",
        "是否拥有",
        "这个道具",
        "这个物品",
        "吗",
        "？",
        "?",
    ]
    for token in replacements:
        text = text.replace(token, "")

    lowered = text.lower()
    lowered = re.sub(r"\bdo i have\b", "", lowered)
    lowered = re.sub(r"\bin my inventory\b", "", lowered)
    lowered = re.sub(r"\bin my backpack\b", "", lowered)
    lowered = re.sub(r"\bthe\b", "", lowered)
    return " ".join(lowered.split()) if lowered.strip() else text.strip()
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
python3 -m pytest tests/test_dialogue_routing.py -v
```

Expected: helper tests pass.

### Task 4: Integrate Router Into the CLI Loop

**Files:**
- Modify: `npc_dialogue.py`

- [ ] **Step 1: Add player profile to main context**

In `main()`, replace `player_context` with:

```python
    player_context = get_player_profile("player_001")
    player_context["quest_status"] = "刚刚抵达小镇，尚未接取任务"
    player_context["reputation"] = "中立"
```

- [ ] **Step 2: Route normal inputs after explicit command handling**

In `chat_with_npc()`, after the `/quest` block and before the persona/RAG response block, add:

```python
        route = classify_intent(player_input)
        print(f"[Router] intent={route.intent} reason={route.reason}")

        if route.intent == QUEST_REQUEST:
            print(f"\n{npc['name']} 沉吟片刻……\n")
            quest_text = generate_quest(npc, player_context, conversation_history)
            print(f"{npc['name']}: {quest_text}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": quest_text})
            continue

        if route.intent == INVENTORY_QUERY:
            item_name = extract_inventory_item_name(player_input)
            inventory_result = check_inventory(player_context, item_name)
            tool_response = format_inventory_response(inventory_result)
            print(f"\n{tool_response}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": tool_response})
            continue

        if route.intent == GM_FEEDBACK:
            feedback_response = (
                "[GM Workflow] 已记录你的反馈。"
                "如果这是任务或剧情问题，GM 会优先检查任务目标、奖励和触发条件是否清晰。"
            )
            if conversation_history:
                feedback_response += " 你也可以输入 quit 结束本轮对话并生成完整 GM 反馈报告。"
            print(f"\n{feedback_response}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": feedback_response})
            continue
```

- [ ] **Step 3: Gate RAG by intent**

Replace:

```python
        active_rag_service = get_rag_service()
```

with:

```python
        active_rag_service = get_rag_service() if should_use_rag_for_intent(route.intent) else None
```

The existing `system_prompt = build_system_prompt(...)` block remains the fallback for `persona_chat` and `lore_question`.

- [ ] **Step 4: Run unit tests**

Run:

```bash
python3 -m pytest tests/test_intent_router.py tests/test_game_tools.py tests/test_dialogue_routing.py -v
```

Expected: all new tests pass.

- [ ] **Step 5: Run full tests and compile**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile npc_dialogue.py prompts.py rag_pipeline.py intent_router.py game_tools.py
```

Expected: all tests pass and compilation succeeds.

### Task 5: Update README for Phase 2

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add router/tool module to overview**

Add to the module list:

```markdown
**Intent Router + Workflow Orchestration** - A deterministic router classifies player input before model calls, sending casual chat, lore questions, quest requests, inventory questions, and GM feedback into separate workflows.

**Local Tool/API Layer** - Inventory and player-profile queries are handled through local Python functions that simulate real backend API calls.
```

- [ ] **Step 2: Add router architecture section**

Add:

```markdown
## Intent Routing

```text
Player input
  -> safety check
  -> explicit commands: quit, /quest
  -> intent_router.classify_intent()
     -> persona_chat: Claude persona prompt without RAG
     -> lore_question: RAG retrieval + Claude persona prompt
     -> quest_request: quest generation workflow
     -> inventory_query: local tool/API response
     -> gm_feedback: GM feedback workflow acknowledgement
```

The router is rule-based in this phase, so it is deterministic, testable, and does not add another LLM call per turn.
```

- [ ] **Step 3: Add examples**

Add:

```markdown
## Router Verification Examples

```text
你: 你今天怎么样？
Router: persona_chat

你: 碎星髓是什么？
Router: lore_question

你: 我想接一个任务
Router: quest_request

你: 我背包里有月盐药剂吗？
Router: inventory_query

你: 我要反馈这个任务不合理
Router: gm_feedback
```
```

- [ ] **Step 4: Update project structure**

Include:

```text
├── intent_router.py         # Rule-based workflow classifier
├── game_tools.py            # Local player profile and inventory tool/API functions
```

- [ ] **Step 5: Run doc sanity check**

Run:

```bash
rg -n "Intent Router|Local Tool|intent_router|game_tools|Router Verification" README.md
```

Expected: all new sections are discoverable.

### Task 6: Final Verification

**Files:**
- No new edits unless verification finds an issue.

- [ ] **Step 1: Run full offline suite**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile npc_dialogue.py prompts.py rag_pipeline.py intent_router.py game_tools.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 2: Verify `quit` does not initialize RAG**

Run:

```bash
printf 'blacksmith\nquit\n' | python3 npc_dialogue.py
```

Expected:
- NPC greeting and farewell print.
- No HuggingFace model loading output.
- No Claude GM feedback call because conversation history is empty.

- [ ] **Step 3: Verify inventory route without Claude**

Run:

```bash
printf 'blacksmith\n我背包里有月盐药剂吗？\nquit\n' | python3 npc_dialogue.py
```

Expected:
- Router prints `intent=inventory_query`.
- Output contains `[工具调用: check_inventory] 你有月盐药剂。`
- `quit` may generate GM feedback because conversation history is non-empty; if no API key is configured, this can fail after proving the tool route. For offline verification, stop before `quit` by testing helper functions instead.

- [ ] **Step 4: Verify lore route initializes RAG**

Run:

```bash
python3 - <<'PY'
from intent_router import classify_intent

for text in ["你今天怎么样？", "碎星髓是什么？", "我想接一个任务", "我背包里有月盐药剂吗？", "我要反馈这个任务不合理"]:
    print(text, "->", classify_intent(text).intent)
PY
```

Expected:

```text
你今天怎么样？ -> persona_chat
碎星髓是什么？ -> lore_question
我想接一个任务 -> quest_request
我背包里有月盐药剂吗？ -> inventory_query
我要反馈这个任务不合理 -> gm_feedback
```

- [ ] **Step 5: Inspect working tree**

Run:

```bash
git status --short -- /Users/shiyize/Desktop/GameNPCDialogue
```

Expected: only intended project files are changed/untracked. Do not stage or commit because Git top-level is `/Users/shiyize`.
