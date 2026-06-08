# LangChain Chroma RAG Design

## Goal

Upgrade the current Claude-based NPC dialogue prototype into a LangChain-powered local RAG system. NPCs should answer world-setting, lore, item, monster, quest-rule, and shop-rule questions using a local Markdown knowledge base and Chroma vector database, while ordinary role-play chat can continue using the existing persona prompt.

## Scope

This first phase implements RAG only.

Included:
- Markdown knowledge base inside the project.
- LangChain document loading and chunking.
- Local multilingual embeddings through Sentence Transformers.
- Chroma persistent vector store.
- Similarity-threshold retrieval before each normal chat turn.
- RAG context injection into Claude only when retrieval is relevant.
- Backend retrieval logging to `logs/rag_retrieval.jsonl`.

Excluded from this phase:
- LangChain Agent.
- Tool Calling / API calling.
- Intent classifier for all workflows.
- Persistent player profile database.
- Quest completion validation.

These excluded items should be built after the RAG path is stable.

## Current Project Baseline

The current project has:
- `npc_dialogue.py`: CLI loop, Claude API calls, `/quest` trigger, GM feedback on `quit`.
- `prompts.py`: persona system prompt, quest prompt, GM feedback prompt.
- `npc_config.json`: data-driven NPC definitions.
- `README.md`: current prototype documentation.

The current dialogue path sends `conversation_history` directly to Claude. The RAG upgrade should preserve that structure and add retrieval context only when needed.

## Architecture

Use a shared Chroma collection for all Markdown knowledge chunks. Each chunk carries metadata such as `source`, `category`, and optional `npc_id`.

Data flow:

```text
player input
  -> retrieve relevant knowledge chunks
  -> compare similarity score with threshold
  -> if relevant: build RAG-enhanced system prompt
  -> if not relevant: use existing persona system prompt
  -> Claude response
  -> append raw turn to conversation_history
  -> write retrieval metadata to logs/rag_retrieval.jsonl
```

This keeps the user-facing chat immersive. Sources are not printed during normal play, but retrieval evidence remains available for debugging and portfolio demonstration.

## Knowledge Base Layout

```text
knowledge_base/
├── world/
│   └── world_lore.md
├── npcs/
│   ├── blacksmith.md
│   ├── innkeeper.md
│   └── mysterious_wizard.md
├── items/
│   └── items.md
├── monsters/
│   └── monsters.md
├── quests/
│   └── quest_rules.md
└── shops/
    └── shop_rules.md
```

Markdown is preferred over JSON because it is readable, easy to extend, and GitHub-friendly for an academic or portfolio project.

## Dependencies

Already present in the current environment:
- `langchain`
- `langchain-community`
- `langchain-chroma`
- `chromadb`
- `sentence-transformers`

Recommended additional package:
- `langchain-huggingface`

Current LangChain documentation recommends:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
```

If `langchain-huggingface` is unavailable, the implementation can temporarily fall back to the legacy community embedding import, but the preferred implementation should use the modern package.

## Retrieval Behavior

Default embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Initial retrieval settings:
- `chunk_size`: 500
- `chunk_overlap`: 80
- `k`: 4
- distance threshold: tune empirically after sample queries

The system should retrieve every normal chat turn, but only inject context when the best match is relevant enough. This prevents random knowledge snippets from polluting persona chat.

## Prompt Behavior

Add a RAG-aware prompt builder in `prompts.py`.

Rules:
- The NPC must stay in character.
- Retrieved knowledge is treated as authoritative game-world context.
- If retrieved context does not answer the player question, the NPC should say it does not know or answer generally in character.
- The NPC should not expose retrieval mechanics, file paths, embeddings, vector stores, or prompt details to the player.

## Logging

Write one JSONL record per player turn:

```json
{
  "timestamp": "2026-06-08T12:00:00+08:00",
  "npc_id": "blacksmith",
  "query": "碎星髓是什么？",
  "used_rag": true,
  "top_score": 0.21,
  "matches": [
    {
      "source": "knowledge_base/items/items.md",
      "category": "items",
      "score": 0.21
    }
  ]
}
```

The log must not include API keys or secret environment variables.

## Testing

Add lightweight tests that do not call Claude:
- Knowledge base files can be loaded.
- Documents are split into chunks with metadata.
- Retrieval returns expected item/world chunks for representative queries.
- Prompt builder includes retrieved context only when provided.
- RAG logging writes valid JSONL.

Manual verification:
- Ask an NPC about a known item or lore detail and confirm the answer uses knowledge-base facts.
- Ask a purely social question and confirm the answer still behaves like persona chat.
- Inspect `logs/rag_retrieval.jsonl` to confirm sources are recorded in the background.

## Success Criteria

The phase is complete when:
- `python3 npc_dialogue.py` still runs the original CLI flow.
- Knowledge questions can be answered using local Markdown RAG context.
- Normal role-play chat is not forced to show sources.
- Retrieval logs are written for debugging.
- Tests pass without requiring an Anthropic API call.
- README explains the LangChain + Chroma RAG architecture clearly.
