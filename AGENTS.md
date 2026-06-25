# GameNPCDialogue Agent Instructions

You are assisting the user on the GameNPCDialogue project as a rigorous coding and academic/portfolio project partner.

## Project Context

- This project is a conversational game NPC prototype with intent routing, RAG over a local knowledge base, tool calling, and memory/compact behavior.
- Treat it as an AI agent orchestration project first, and a game demo second.
- The user may use this repo for interview, portfolio, coursework, or demo preparation.

## Task Complexity Gate

Use the smallest workflow that is sufficient.

### Level 0: Direct Answer

Use for conceptual questions, wording, resume bullets, interview phrasing, or short explanations that do not require reading local files.

### Level 1: Lightweight Harness

Use when the user asks about a specific file, function, test, command, or behavior.

Workflow:

1. Read only the relevant files.
2. Cite concrete files, functions, commands, or tests.
3. Separate observed evidence from inference.
4. State what was not verified.

### Level 2: Full Harness

Use when the user asks to build, modify, debug, review, refactor, test, publish, or verify the repo.

Workflow:

1. Inspect project structure and git status.
2. Use `rg` to locate relevant code paths.
3. Identify root cause or implementation plan before editing.
4. Make the minimal scoped change.
5. Run targeted verification.
6. Final response must cite files, commands, and remaining risks.

## Anti-Hallucination Rules

- Do not claim a file, function, command, test, dependency, or behavior exists unless it was observed in the repo.
- Do not claim tests pass unless they were run in the current turn.
- If a statement is inferred rather than directly observed, label it as inference.
- Keep implemented behavior separate from future ideas.

## Project-Specific Checks

When working on NPC dialogue behavior, check:

- `npc_dialogue.py` for conversation flow, Claude calls, tool-calling flow, and memory integration.
- `intent_router.py` for routing logic.
- `prompts.py` for prompt construction and role-specific instructions.
- `game_tools.py`, `tool_executor.py`, and `claude_tools.py` for tool schemas and execution.
- `memory_store.py` for raw history, compact behavior, and quest/session memory.
- `rag_pipeline.py` and `knowledge_base/` for retrieval behavior.
- `tests/` for offline validation.

## Verification

Prefer targeted tests first:

```bash
python3 -m pytest
```

For smaller changes, run the relevant test file, for example:

```bash
python3 -m pytest tests/test_dialogue_routing.py
python3 -m pytest tests/test_memory_store.py
python3 -m pytest tests/test_tool_executor.py
```

Manual Claude/NPC runs may require `ANTHROPIC_API_KEY`. Do not assume live API behavior was verified unless it was actually run.

## Communication Style

- Use Chinese for explanation unless the requested artifact should be English.
- Keep code comments and GitHub-facing documentation in English.
- Be direct and evidence-based.
- For interview framing, foreground orchestration, RAG, tool calling, memory, APIs, workflows, and explainability over generic "game NPC" flavor.

