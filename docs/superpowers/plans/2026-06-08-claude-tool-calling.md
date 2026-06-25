# Claude Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Claude native tool calling so tool-worthy NPC inputs can trigger Anthropic `tool_use`, execute local game APIs, return `tool_result`, and produce a final Claude response.

**Architecture:** `claude_tools.py` centralizes Anthropic-compatible tool schemas. `tool_executor.py` executes Claude `tool_use` blocks against local `game_tools.py` functions and returns JSON-serializable results plus `tool_result` content blocks. `npc_dialogue.py` uses tool calling only for tool-worthy intents, while existing `/quest`, RAG lore, persona chat, and GM feedback paths remain intact.

**Tech Stack:** Python 3.13, Anthropic Messages API, pytest, existing local `game_tools.py`, existing router/RAG modules

---

## File Map

- Create `claude_tools.py`: Anthropic `tools` list and lookup helpers.
- Create `tool_executor.py`: fake/real Claude `tool_use` parsing, local execution, `tool_result` block construction.
- Create `tests/test_claude_tools.py`: schema tests.
- Create `tests/test_tool_executor.py`: offline executor tests.
- Modify `game_tools.py`: add `generate_quest_metadata()` and `validate_quest_completion()` local API functions.
- Modify `tests/test_game_tools.py`: cover new local APIs.
- Modify `npc_dialogue.py`: add a tool-calling handler for `inventory_query` and next-action / validation style requests.
- Modify `README.md`: document Claude tool calling architecture and manual verification.

### Task 1: Add Local API Functions Needed by Tool Calling

**Files:**
- Modify: `game_tools.py`
- Modify: `tests/test_game_tools.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_game_tools.py`:

```python
from game_tools import generate_quest_metadata, validate_quest_completion


def test_generate_quest_metadata_returns_level_aware_payload():
    result = generate_quest_metadata("blacksmith", 5)

    assert result["npc_id"] == "blacksmith"
    assert result["player_level"] == 5
    assert result["recommended_type"]
    assert result["difficulty"] in {"low", "medium", "high"}


def test_validate_quest_completion_requires_known_evidence():
    incomplete = validate_quest_completion("player_001", "dragon_hunter_weapon")
    complete = validate_quest_completion(
        "player_001",
        "dragon_hunter_weapon",
        evidence=["裂鳞长枪"],
    )

    assert incomplete["completed"] is False
    assert complete["completed"] is True
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_game_tools.py -v
```

Expected: import failure for missing functions.

- [ ] **Step 3: Implement local API functions**

Append to `game_tools.py`:

```python
QUEST_REQUIREMENTS = {
    "dragon_hunter_weapon": ["裂鳞长枪"],
    "moon_salt_potion": ["月盐药剂"],
    "star_marrow_sample": ["碎星髓"],
}


def generate_quest_metadata(npc_id: str, player_level: int) -> dict:
    quest_type_by_npc = {
        "blacksmith": "收集魔龙猎人武器",
        "innkeeper": "调查失踪旅客情报",
        "mysterious_wizard": "收集安全碎星髓样本",
    }
    difficulty = "low"
    if player_level >= 8:
        difficulty = "high"
    elif player_level >= 5:
        difficulty = "medium"

    return {
        "npc_id": npc_id,
        "player_level": player_level,
        "recommended_type": quest_type_by_npc.get(npc_id, "外围调查"),
        "difficulty": difficulty,
    }


def validate_quest_completion(
    player_id: str,
    quest_id: str,
    evidence: list[str] | None = None,
) -> dict:
    required_evidence = QUEST_REQUIREMENTS.get(quest_id, [])
    provided_evidence = evidence or []
    completed = bool(required_evidence) and all(
        required in provided_evidence for required in required_evidence
    )
    return {
        "player_id": player_id,
        "quest_id": quest_id,
        "completed": completed,
        "required_evidence": list(required_evidence),
        "provided_evidence": list(provided_evidence),
    }
```

- [ ] **Step 4: Run local API tests**

Run:

```bash
python3 -m pytest tests/test_game_tools.py -v
```

Expected: all game tool tests pass.

### Task 2: Define Claude Tool Schemas

**Files:**
- Create: `claude_tools.py`
- Create: `tests/test_claude_tools.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_claude_tools.py`:

```python
from claude_tools import CLAUDE_TOOLS, get_tool_schema


EXPECTED_TOOLS = {
    "get_player_profile",
    "check_inventory",
    "recommend_next_action",
    "generate_quest",
    "validate_quest_completion",
}


def test_claude_tools_include_expected_names():
    names = {tool["name"] for tool in CLAUDE_TOOLS}

    assert names == EXPECTED_TOOLS


def test_each_tool_has_anthropic_required_fields():
    for tool in CLAUDE_TOOLS:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["description"]
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]


def test_get_tool_schema_returns_matching_schema():
    schema = get_tool_schema("check_inventory")

    assert schema["name"] == "check_inventory"
    assert "item_name" in schema["input_schema"]["properties"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_claude_tools.py -v
```

Expected: collection failure for missing `claude_tools`.

- [ ] **Step 3: Implement tool schemas**

Create `claude_tools.py`:

```python
from __future__ import annotations


CLAUDE_TOOLS = [
    {
        "name": "get_player_profile",
        "description": "Read the player's current level, class, inventory, and active quest.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string", "description": "Player identifier."},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "check_inventory",
        "description": "Check whether the player owns a specific item.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "item_name": {"type": "string"},
            },
            "required": ["player_id", "item_name"],
        },
    },
    {
        "name": "recommend_next_action",
        "description": "Recommend what the player should do next based on player state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "generate_quest",
        "description": "Generate quest metadata based on NPC and player level. This does not replace the narrative quest generator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "npc_id": {"type": "string"},
                "player_level": {"type": "integer"},
            },
            "required": ["npc_id", "player_level"],
        },
    },
    {
        "name": "validate_quest_completion",
        "description": "Validate whether a quest is completed using explicit evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "quest_id": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence items the player provides.",
                },
            },
            "required": ["player_id", "quest_id"],
        },
    },
]


def get_tool_schema(tool_name: str) -> dict | None:
    for tool in CLAUDE_TOOLS:
        if tool["name"] == tool_name:
            return tool
    return None
```

- [ ] **Step 4: Run schema tests**

Run:

```bash
python3 -m pytest tests/test_claude_tools.py -v
```

Expected: schema tests pass.

### Task 3: Implement Tool Executor

**Files:**
- Create: `tool_executor.py`
- Create: `tests/test_tool_executor.py`

- [ ] **Step 1: Write failing executor tests**

Create `tests/test_tool_executor.py`:

```python
from tool_executor import execute_tool_call, tool_result_block


def test_execute_tool_call_runs_check_inventory():
    result = execute_tool_call(
        {
            "name": "check_inventory",
            "input": {"player_id": "player_001", "item_name": "月盐药剂"},
        }
    )

    assert result["ok"] is True
    assert result["tool_name"] == "check_inventory"
    assert result["result"]["has_item"] is True


def test_execute_tool_call_reports_unknown_tool():
    result = execute_tool_call({"name": "unknown_tool", "input": {}})

    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_execute_tool_call_reports_missing_argument():
    result = execute_tool_call({"name": "check_inventory", "input": {"player_id": "player_001"}})

    assert result["ok"] is False
    assert "missing required argument" in result["error"]


def test_tool_result_block_serializes_result():
    execution = execute_tool_call(
        {
            "name": "check_inventory",
            "input": {"player_id": "player_001", "item_name": "月盐药剂"},
        }
    )

    block = tool_result_block("toolu_123", execution)

    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_123"
    assert "月盐药剂" in block["content"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_tool_executor.py -v
```

Expected: collection failure for missing `tool_executor`.

- [ ] **Step 3: Implement executor**

Create `tool_executor.py`:

```python
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from game_tools import (
    check_inventory,
    generate_quest_metadata,
    get_player_profile,
    recommend_next_action,
    validate_quest_completion,
)


def _check_inventory_tool(player_id: str, item_name: str) -> dict:
    profile = get_player_profile(player_id)
    return check_inventory(profile, item_name)


def _recommend_next_action_tool(player_id: str) -> dict:
    profile = get_player_profile(player_id)
    return {
        "player_id": player_id,
        "recommendation": recommend_next_action(profile),
    }


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_player_profile": get_player_profile,
    "check_inventory": _check_inventory_tool,
    "recommend_next_action": _recommend_next_action_tool,
    "generate_quest": generate_quest_metadata,
    "validate_quest_completion": validate_quest_completion,
}


def execute_tool_call(tool_call: dict) -> dict:
    tool_name = tool_call.get("name")
    tool_input = tool_call.get("input") or {}

    if tool_name not in TOOL_FUNCTIONS:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"unknown tool: {tool_name}",
        }

    function = TOOL_FUNCTIONS[tool_name]
    signature = inspect.signature(function)
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and name not in tool_input
    ]
    if missing:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": f"missing required argument: {missing[0]}",
        }

    try:
        result = function(**tool_input)
    except Exception as exc:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": str(exc),
        }

    return {
        "ok": True,
        "tool_name": tool_name,
        "result": result,
    }


def tool_result_block(tool_use_id: str, execution_result: dict) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(execution_result, ensure_ascii=False),
    }
```

- [ ] **Step 4: Run executor tests**

Run:

```bash
python3 -m pytest tests/test_tool_executor.py -v
```

Expected: executor tests pass.

### Task 4: Integrate Claude Tool Calling Into CLI

**Files:**
- Modify: `npc_dialogue.py`
- Create: `tests/test_dialogue_tool_calling.py`

- [ ] **Step 1: Add helper tests without calling Claude**

Create `tests/test_dialogue_tool_calling.py`:

```python
from npc_dialogue import should_use_claude_tools


def test_should_use_claude_tools_for_tool_worthy_intents():
    assert should_use_claude_tools("inventory_query") is True
    assert should_use_claude_tools("persona_chat") is False
    assert should_use_claude_tools("lore_question") is False
    assert should_use_claude_tools("quest_request") is False
    assert should_use_claude_tools("gm_feedback") is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_dialogue_tool_calling.py -v
```

Expected: import failure for missing helper.

- [ ] **Step 3: Add imports and helper**

Modify `npc_dialogue.py` imports:

```python
from claude_tools import CLAUDE_TOOLS
from tool_executor import execute_tool_call, tool_result_block
```

Add helper after `should_use_rag_for_intent()`:

```python
def should_use_claude_tools(intent: str) -> bool:
    return intent == INVENTORY_QUERY
```

- [ ] **Step 4: Add tool-calling handler**

Add this function before `chat_with_npc()`:

```python
def run_claude_tool_turn(npc: dict, player_context: dict, player_input: str) -> str:
    system_prompt = build_system_prompt(npc, player_context)
    messages = [{"role": "user", "content": player_input}]
    first_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=messages,
        tools=CLAUDE_TOOLS,
    )

    tool_uses = [
        block for block in first_response.content
        if getattr(block, "type", None) == "tool_use"
    ]
    if not tool_uses:
        return first_response.content[0].text

    tool_results = []
    for tool_use in tool_uses:
        execution = execute_tool_call({
            "name": tool_use.name,
            "input": tool_use.input,
        })
        tool_results.append(tool_result_block(tool_use.id, execution))

    follow_up = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[
            messages[0],
            {"role": "assistant", "content": first_response.content},
            {"role": "user", "content": tool_results},
        ],
    )
    return follow_up.content[0].text
```

- [ ] **Step 5: Replace direct inventory branch with Claude tool branch**

In `chat_with_npc()`, replace the `INVENTORY_QUERY` deterministic branch with:

```python
        if should_use_claude_tools(route.intent):
            try:
                tool_reply = run_claude_tool_turn(npc, player_context, player_input)
            except Exception as exc:
                item_name = extract_inventory_item_name(player_input)
                inventory_result = check_inventory(player_context, item_name)
                tool_reply = format_inventory_response(inventory_result)
                print(f"[Tool Calling 警告] Claude 工具调用失败，已使用本地工具兜底：{exc}\n")
            print(f"\n{npc['name']}: {tool_reply}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": tool_reply})
            continue
```

This preserves offline fallback and avoids breaking CLI if API key/tool call fails.

- [ ] **Step 6: Run tests**

Run:

```bash
python3 -m pytest tests/test_dialogue_tool_calling.py -v
python3 -m pytest -q
python3 -m py_compile npc_dialogue.py claude_tools.py tool_executor.py game_tools.py
```

Expected: all tests pass and compile succeeds.

### Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add tool calling module to overview**

Add:

```markdown
**Claude Tool Calling Layer** - Tool-worthy player requests can be sent to Claude with Anthropic tool schemas. Claude can request a tool, the local executor runs the matching Python API, and the result is returned to Claude for the final NPC response.
```

- [ ] **Step 2: Add architecture section**

Add:

```markdown
## Claude Tool Calling

```text
Player tool-worthy input
  -> Claude request with CLAUDE_TOOLS
  -> Claude returns tool_use
  -> tool_executor.execute_tool_call()
  -> user message with tool_result
  -> Claude final NPC response
```

Offline tests validate schemas and execution without calling Claude. Manual tests require `ANTHROPIC_API_KEY`.
```

- [ ] **Step 3: Update project structure**

Add:

```text
├── claude_tools.py          # Anthropic tool schemas
├── tool_executor.py         # Executes Claude tool_use blocks locally
```

- [ ] **Step 4: Add manual verification**

Add:

```markdown
Manual tool-calling check:

```text
你: 我背包里有月盐药剂吗？
```

Expected behavior: Claude uses `check_inventory`, the local executor returns the inventory result, and the final NPC response reflects that result.
```

- [ ] **Step 5: Sanity check README**

Run:

```bash
rg -n "Claude Tool Calling|claude_tools|tool_executor|tool_use|tool_result" README.md
```

Expected: all new terms appear.

### Task 6: Final Verification

**Files:**
- No new edits unless verification finds an issue.

- [ ] **Step 1: Run full offline checks**

Run:

```bash
python3 -m pytest -q
python3 -m py_compile npc_dialogue.py prompts.py rag_pipeline.py intent_router.py game_tools.py claude_tools.py tool_executor.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 2: Run quick CLI smoke without Claude**

Run:

```bash
printf 'blacksmith\nquit\n' | python3 npc_dialogue.py
```

Expected: greeting and farewell print, no HuggingFace model loading, no Claude GM feedback because history is empty.

- [ ] **Step 3: Run offline executor demo**

Run:

```bash
python3 - <<'PY'
from tool_executor import execute_tool_call

print(execute_tool_call({
    "name": "check_inventory",
    "input": {"player_id": "player_001", "item_name": "月盐药剂"},
}))
PY
```

Expected: output contains `ok: True` and `has_item: True`.

- [ ] **Step 4: Optional manual Claude test**

Only if `ANTHROPIC_API_KEY` is configured:

```bash
python3 npc_dialogue.py
```

Manual input:

```text
blacksmith
我背包里有月盐药剂吗？
quit
```

Expected: tool-calling branch runs. If Claude tool calling fails, local fallback should still answer inventory deterministically.
