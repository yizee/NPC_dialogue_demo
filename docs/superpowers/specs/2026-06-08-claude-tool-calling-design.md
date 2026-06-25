# Claude Tool Calling Design

## Goal

Upgrade Phase 2's local tool/API simulation into Claude native tool calling. The NPC should be able to ask Claude to select a tool, execute the matching local Python function, send the tool result back to Claude, and then produce a final in-character response.

This phase is designed to demonstrate API integration and tool orchestration for Solution Engineer / AI Engineer portfolio use.

## Scope

Included:
- Define Anthropic-compatible tool schemas with `name`, `description`, and `input_schema`.
- Execute Claude `tool_use` blocks through a local tool executor.
- Return `tool_result` blocks to Claude using the Messages API flow.
- Add tool-calling support for inventory, player profile, next action, quest generation metadata, and quest completion validation.
- Add offline tests for tool schema and executor behavior.
- Keep existing deterministic router, RAG, persona chat, `/quest`, and GM feedback behavior.

Excluded:
- LangChain Agent.
- Real database or external game backend.
- Persistent quest state machine.
- Full true-ending / Goblin King story state.
- Streaming tool calls.

## Tools

Expose these Claude tools:

```text
get_player_profile(player_id)
check_inventory(player_id, item_name)
recommend_next_action(player_id)
generate_quest(npc_id, player_level)
validate_quest_completion(player_id, quest_id)
```

`game_tools.py` remains the local API implementation layer. Phase 3 may extend it with missing functions, but tool schemas and execution orchestration should live outside `npc_dialogue.py`.

## Proposed Files

```text
claude_tools.py
tool_executor.py
tests/test_claude_tools.py
tests/test_tool_executor.py
```

Responsibilities:
- `claude_tools.py`: Anthropic `tools` list and helper lookup by tool name.
- `tool_executor.py`: translate Claude `tool_use` blocks into local function calls and return JSON-serializable tool results.
- `game_tools.py`: local game API functions only.
- `npc_dialogue.py`: conversation orchestration only.

## Anthropic Message Flow

For tool-worthy intents:

```text
user input
  -> Claude request with tools=[...]
  -> assistant response may contain tool_use
  -> local executor runs matching tool
  -> send user message with tool_result
  -> Claude final response
```

The implementation should support one tool call per turn for this phase. If Claude returns multiple `tool_use` blocks, execute them sequentially and return all matching `tool_result` blocks in one follow-up user message.

## Integration Behavior

Use Claude tool calling for:
- `inventory_query`
- next-action style requests
- quest completion validation style requests

Keep existing direct paths for:
- `/quest`
- `quest_request` natural language quest generation
- `lore_question` RAG path
- `persona_chat`
- `gm_feedback`

This avoids overloading Phase 3 and keeps quest story state for Phase 4.

## Error Handling

The tool executor should return structured error results instead of crashing:

```json
{
  "ok": false,
  "tool_name": "check_inventory",
  "error": "missing required argument: item_name"
}
```

Unknown tool names should also return `ok: false`.

Claude-facing final text should not expose Python stack traces.

## Testing

Offline tests must cover:
- Tool schemas contain required Anthropic fields.
- Each schema has a valid object `input_schema`.
- `execute_tool_call()` runs known tools.
- Missing arguments return structured errors.
- Unknown tool names return structured errors.
- Fake Claude `tool_use` blocks can be converted into `tool_result` blocks.

Manual verification:
- Ask `我背包里有月盐药剂吗？`
- Confirm Claude chooses or is forced to use `check_inventory`.
- Confirm final response mentions the tool result in character or clean CLI style.

## Success Criteria

- `python3 -m pytest -q` passes.
- Existing Phase 1 and Phase 2 behavior is preserved.
- Tool schemas are centralized in `claude_tools.py`.
- Tool execution is centralized in `tool_executor.py`.
- Tool tests run without calling Claude.
- Manual Claude test can complete at least one tool-use round trip.
