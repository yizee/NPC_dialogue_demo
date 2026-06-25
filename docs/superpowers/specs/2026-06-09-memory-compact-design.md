# Memory Compact Design

## Goal

Add a persistent dialogue memory layer so NPC conversations can continue with stable context across long sessions. The system should preserve every raw player/NPC turn for audit and story generation, while passing only recent and compressed memory into Claude during live dialogue.

This phase supports the project's core design goal: NPCs should feel like flexible characters inside a consistent world, not rigid dialogue trees or stateless LLM wrappers.

## Scope

Included:
- Persist raw conversation turns to session JSONL logs.
- Maintain a recent conversation window for live Claude calls.
- Define a compact summary structure for long conversations.
- Track story events separately from raw dialogue.
- Track quest state separately from compact summaries.
- Add offline tests for memory storage, recent-window trimming, compact trigger logic, and quest-state updates.
- Integrate the memory layer into `npc_dialogue.py` without changing the existing router, RAG, tool calling, or quest generation behavior.

Excluded:
- Final novel generation.
- Full ending-condition engine.
- Multi-save UI.
- Real database storage.
- Long-term cross-session NPC relationship graph.
- Claude-based automatic compact generation as the only implementation path.

Phase 4 may define the Claude compact prompt contract, but the first implementation should keep memory storage testable without making API calls.

## Core Principle

Compact is a memory compression layer, not a deletion mechanism.

The system must keep:

```text
raw_conversation_log: full, append-only record
recent_conversation_history: short window for live replies
session_summary: compressed memory for long context
story_events: cleaned plot-relevant events
quest_state: structured task state
```

Raw logs are used for audit, debugging, replay, and future final-story generation. Compact summaries are used to keep Claude's prompt short and focused during live NPC dialogue.

## Proposed Files

```text
memory_store.py
tests/test_memory_store.py
```

Responsibilities:
- `memory_store.py`: session creation, JSONL append, recent-window trimming, compact trigger checks, story event storage, and quest-state storage.
- `prompts.py`: compact prompt builder and output contract only.
- `npc_dialogue.py`: orchestration. It should call memory functions but should not own memory data structures.
- `game_tools.py`: remains the local game API layer.
- `tool_executor.py`: remains the Claude tool execution layer.

## Memory Data Model

Use JSON-serializable dictionaries so logs are easy to inspect.

### Raw Turn

Each saved turn should include:

```json
{
  "session_id": "20260609_153012_mysterious_wizard_player_001",
  "turn_index": 3,
  "timestamp": "2026-06-09T15:30:12+08:00",
  "npc_id": "mysterious_wizard",
  "player_id": "player_001",
  "user_input": "我背包里有月盐药剂吗？",
  "assistant_reply": "你确实带着一瓶月盐药剂...",
  "intent": "inventory_query",
  "used_rag": false,
  "used_tool_calling": true,
  "rag_sources": [],
  "tool_names": ["check_inventory"]
}
```

### Runtime Memory State

The in-memory session object should expose:

```json
{
  "session_id": "...",
  "recent_conversation_history": [],
  "session_summary": "",
  "story_events": [],
  "quest_state": {
    "active_quest": null,
    "completed_quests": [],
    "failed_quests": [],
    "known_clues": [],
    "pending_commitments": []
  }
}
```

`recent_conversation_history` should use the same role/content shape expected by Anthropic:

```json
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
```

## Compact Output Contract

When compact is triggered, the summary should be structured:

```json
{
  "session_summary": "玩家与神秘法师建立了初步信任...",
  "important_facts": [
    "玩家知道月盐药剂可以提升魔力感知。"
  ],
  "quest_progress": [
    "玩家尚未正式选择职业，但表现出学习魔法的兴趣。"
  ],
  "player_commitments": [
    "玩家承诺会寻找第一本入门魔法书。"
  ],
  "npc_attitude": {
    "mysterious_wizard": "谨慎但愿意继续观察玩家。"
  },
  "story_events": [
    {
      "event_type": "clue_revealed",
      "summary": "神秘法师暗示魔法书可能藏在旧塔。"
    }
  ],
  "open_threads": [
    "玩家是否会选择法师职业仍未确定。"
  ]
}
```

The compact result should preserve:
- Important information the player already knows.
- Clues already revealed by NPCs.
- Current quest progress.
- Player commitments.
- NPC attitude changes.
- Plot-relevant world-state changes.

The compact result should remove:
- Repeated greetings.
- Small talk with no story impact.
- Duplicate wording.
- Details that do not affect quests, relationships, lore, or future choices.

## Compact Trigger Rules

Use deterministic trigger logic in Python:

```text
trigger compact when:
- recent_conversation_history exceeds max_recent_messages
- or raw turn count reaches compact_every_n_turns
- or the user exits with quit
```

Recommended MVP defaults:

```text
max_recent_messages = 12
compact_every_n_turns = 8
```

These values are small enough for testing and demonstration. Phase 4 MVP should keep them as module-level defaults; moving them into a user-facing config is outside this phase.

## Chat Flow Integration

The live chat flow should become:

```text
player input
  -> intent_router.classify_intent()
  -> selected workflow
     -> persona chat
     -> RAG retrieval
     -> quest workflow
     -> Claude tool calling
     -> GM feedback
  -> NPC response
  -> append raw turn to JSONL
  -> update recent_conversation_history
  -> update story_events / quest_state when applicable
  -> compact if trigger condition is met
```

Claude live prompts should use:

```text
NPC persona
player profile
quest_state
session_summary
recent_conversation_history
RAG context when intent == lore_question
tool result when tool calling is used
```

Claude should not receive the full raw log every turn.

## Quest State Boundary

Compact may summarize quest-relevant conversation, but it must not decide quest completion.

Allowed:
- "玩家答应去寻找魔龙猎人的武器。"
- "NPC 已经透露黑铁夹具与魔法自修炼有关。"

Not allowed:
- Marking a quest complete only because the summary sounds complete.
- Unlocking the true ending only from a compact paragraph.

Quest completion should be validated by explicit game state, inventory evidence, or `validate_quest_completion()`.

## Error Handling

Memory writes should fail softly during CLI demo:
- If JSONL append fails, print a concise warning and continue the conversation.
- If compact generation fails, preserve raw logs and keep recent history.
- If a compact result is malformed, ignore that compact result and keep the previous summary.

The system should never lose raw conversation because a compact step failed.

## Testing

Offline tests should cover:
- Creating a session id.
- Appending raw turns to JSONL.
- Maintaining recent history with a fixed max size.
- Compact trigger behavior by message count and turn count.
- Updating story events.
- Updating quest state without relying on compact.
- Building compact input from summary plus recent messages.
- Ensuring raw logs are still written when compact is skipped.

Manual verification:
- Start a CLI session and chat for more than the recent-window size.
- Confirm `logs/sessions/*.jsonl` contains full raw turns.
- Confirm only recent turns are sent into normal Claude dialogue.
- Confirm `quit` can still generate GM feedback.
- Confirm existing RAG and tool-calling tests still pass.

## Success Criteria

- `python3 -m pytest -q` passes.
- Existing Phase 1 RAG behavior is preserved.
- Existing Phase 2 intent routing behavior is preserved.
- Existing Phase 3 Claude tool-calling behavior is preserved.
- Every valid player/NPC exchange is appended to a raw JSONL session log.
- Live Claude calls no longer depend on passing the full conversation history.
- Compact summary structure is defined and testable.
- Quest state remains separate from compact summary text.
