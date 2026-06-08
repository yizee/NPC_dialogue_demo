# Intent Router Workflow Design

## Goal

Add a deterministic intent router so the NPC system can route player input to the right workflow before calling Claude. This upgrades the project from a RAG-enabled chat prototype into a clearer AI orchestration demo.

## Scope

This phase implements a rule-based MVP.

Included:
- Classify ordinary player input into workflow intents.
- Route lore/world-setting questions through the existing RAG-enhanced persona chat path.
- Route casual chat through persona chat without forcing RAG.
- Route natural task requests to the existing quest generator, in addition to `/quest`.
- Route item/inventory questions to local tool-like functions.
- Route complaint/feedback messages to a lightweight GM feedback workflow.
- Add offline tests for routing and tool behavior.

Excluded:
- Claude-based intent classification.
- LangChain Agent.
- Real database or external API integration.
- Persistent quest state.
- Quest completion validation.

## Intents

Use these intent labels:

```text
persona_chat
lore_question
quest_request
inventory_query
gm_feedback
```

Fallback behavior must be `persona_chat`.

## Routing Rules

Rules are deterministic and keyword-based. They should support both Chinese and common English terms.

Recommended priority:

1. `gm_feedback`: complaint, bug, feedback, report, 投诉, 反馈, 举报, 卡住, 不合理
2. `inventory_query`: backpack, inventory, item, have, 背包, 物品, 道具, 有没有, 是否拥有
3. `quest_request`: quest, mission, task, 任务, 委托, 接任务, 有什么事要我做
4. `lore_question`: who, what, where, why, lore, world, monster, item explanation, 是什么, 谁是, 在哪里, 为什么, 世界观, 魔龙, 地精之王
5. `persona_chat`: default

The router does not call Claude. It should return a small structured object containing `intent`, `confidence`, and `reason`.

## Tool-like Local API

Add a small local tool module to demonstrate API-style integration:

```python
get_player_profile(player_id)
check_inventory(player_context, item_name)
recommend_next_action(player_context)
```

For this phase, use local in-memory dictionaries. Do not add persistence.

Expected default player state:
- name: 旅行者
- level: 5
- class: 未选择
- inventory: 黑铁夹具, 月盐药剂
- active_quest: None

## Chat Flow

```text
player input
  -> safety check
  -> explicit commands
     -> quit
     -> /quest
  -> intent router
     -> quest_request: generate_quest()
     -> inventory_query: call game_tools and print result
     -> gm_feedback: generate short local acknowledgement or GM summary if history exists
     -> lore_question: use RAG-enhanced persona chat
     -> persona_chat: persona chat with RAG disabled for that turn
```

This intentionally changes the current behavior where all ordinary chat attempts RAG first. After Phase 2, RAG should be reserved for likely lore questions, reducing irrelevant retrieval and improving latency.

## Prompt Behavior

No new Claude prompt is required for the router. Existing prompt builders remain valid:
- `build_system_prompt()` handles persona and RAG context.
- `build_quest_prompt()` handles quest generation.
- `build_gm_prompt()` handles full-session GM feedback.

Inventory/tool answers should be deterministic local text, not Claude-generated.

## Testing

Add offline tests:
- Chinese and English examples classify into each intent.
- Fallback goes to `persona_chat`.
- Inventory tool checks known and missing items.
- `recommend_next_action()` returns a useful recommendation based on class, level, active quest, and inventory.
- A small routing helper can be tested without calling Anthropic.

Manual verification:
- `碎星髓是什么？` routes to `lore_question` and can use RAG.
- `你今天怎么样？` routes to `persona_chat`.
- `我想接一个任务` routes to `quest_request`.
- `我背包里有月盐药剂吗？` routes to `inventory_query`.
- `我要反馈这个任务不合理` routes to `gm_feedback`.

## Success Criteria

- `python3 -m pytest -q` passes.
- `python3 npc_dialogue.py` still supports `quit` and `/quest`.
- Natural task requests can trigger the quest flow.
- Item ownership questions return local tool output without Claude.
- Lore questions still use the existing RAG path.
- Casual chat does not initialize RAG unless classified as `lore_question`.
