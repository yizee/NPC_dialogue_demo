# Game NPC Dialogue System

A prototype AI-powered NPC dialogue and quest generation system built with the Anthropic Claude API, LangChain, and Chroma. It explores how large language models can support dynamic game narrative design while grounding NPC answers in a local Markdown knowledge base.

---

## Overview

This project demonstrates how prompt engineering, retrieval-augmented generation, and LLM APIs can create immersive, context-aware NPC interactions without a traditional rule-based dialogue tree. Each NPC maintains a consistent personality, responds dynamically to player context, can generate quests on demand, and can reference local world knowledge when relevant.

The system combines six functional modules:

**NPC Dialogue (方案 A)** - Multi-turn conversations with distinct NPC personalities, each driven by a structured system prompt that defines character, speech style, backstory, and topic constraints.

**Quest Generator (方案 B)** - A `/quest` command triggers in-character quest generation that references the current conversation history, producing narrative-coherent tasks with item lore and rewards that match each NPC's identity.

**GM Feedback Tool (方案 C)** - After each session, a GM-perspective summary is automatically generated, analyzing player engagement, quest intent, narrative immersion, and providing design suggestions for NPC optimization.

**Local RAG Knowledge System** - Lore and world-setting questions can retrieve relevant Markdown lore from a local knowledge base using LangChain, multilingual embeddings, and Chroma similarity search. Retrieved context is injected into the NPC prompt so answers can reference consistent world, item, monster, quest, and shop facts.

**Intent Router + Workflow Orchestration** - A deterministic router classifies player input before model calls, sending casual chat, lore questions, quest requests, inventory questions, and GM feedback into separate workflows.

**Local Tool/API Layer** - Inventory and player-profile queries are handled through local Python functions that simulate real backend API calls.

---

## RAG Architecture

```text
Player message
    |
    v
LangChain document retriever
    |
    v
Local multilingual embedding model
    |
    v
Chroma similarity search in chroma_db/
    |
    v
Score threshold filtering
    |
    v
Retrieved context injection into NPC prompt
    |
    v
Claude NPC response
    |
    v
JSONL retrieval log in logs/rag_retrieval.jsonl
```

`rag_pipeline.py` owns Markdown loading, metadata handling, chunking, Chroma indexing, threshold-based retrieval, and JSONL logging. `npc_dialogue.py` initializes RAG lazily only for inputs classified as lore questions. `prompts.py` injects retrieved context into the persona prompt without exposing raw source metadata to the player.

The RAG system is local-first: embeddings run locally with a multilingual sentence-transformer model, and Chroma persists vector data under `chroma_db/`.

---

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

---

## Demo

```text
=== NPC 对话 + 任务生成原型 ===
选择 NPC: mysterious_wizard

你遇到了：艾尔文 - 隐居的老法师
艾尔文: ……你来了。星辰早就告诉我，会有一个……有趣的旅行者经过。

你: 碎星髓是什么？

艾尔文: 碎星髓是星落事件后出现在霜牙沼泽的银蓝色矿物。
        它会保存附近生物短暂而破碎的记忆，徒手接触可能带来幻觉。

你: /quest

艾尔文: 三日前，北边的霜牙沼泽有什么东西落下去了……
        那里沉着一块「碎星髓」，把它带回来，我便将书页里的一个秘密送予你。

[GM 反馈报告]
对话参与度：玩家主动探索了物品 lore 和任务线索，互动积极。
任务意向：玩家通过 /quest 明确表达了接取任务的意愿。
角色沉浸感：输入符合奇幻世界观，无出戏行为。
GM 建议：可围绕碎星髓的记忆残响设计后续任务链。
```

---

## Project Structure

```text
├── npc_dialogue.py          # Main program: conversation loop, router, RAG, quest and GM triggers
├── intent_router.py         # Rule-based workflow classifier
├── game_tools.py            # Local player profile and inventory tool/API functions
├── prompts.py               # Prompt builders for persona chat, quest generation, and GM feedback
├── rag_pipeline.py          # LangChain + Chroma RAG loading, indexing, retrieval, and logging
├── npc_config.json          # NPC character definitions
├── knowledge_base/          # Local Markdown lore used by RAG
│   ├── items/
│   ├── monsters/
│   ├── npcs/
│   ├── quests/
│   ├── shops/
│   └── world/
├── tests/                   # RAG and prompt behavior tests
├── requirements.txt         # Runtime and test dependencies
├── pytest.ini               # Pytest configuration
└── README.md
```

Generated local runtime data:

```text
├── chroma_db/               # Local Chroma vector store, created on first indexing
└── logs/rag_retrieval.jsonl # Retrieval audit log
```

---

## Getting Started

**1. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
python3 -m pip install -r requirements.txt
```

**3. Configure the Anthropic API key**

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

You can also store the key in a local `.env` file if your environment loads it automatically.

**4. Run the dialogue prototype**

```bash
python3 npc_dialogue.py
```

The first normal knowledge query may download the local embedding model. Chroma vector data is persisted to `chroma_db/`, so later runs can reuse the local index.

**Available commands during dialogue**

| Command | Action |
|--------|--------|
| `[text]` | Talk to the NPC |
| `/quest` | Request a quest from the NPC |
| `quit` | End session and generate GM feedback report |

---

## RAG Verification Examples

After starting the program, choose `mysterious_wizard` and ask a knowledge question:

```text
你: 碎星髓是什么？
```

Expected behavior: the answer should mention local lore such as silver-blue ore, the Starfall event, Frostfang Marsh, memory fragments, or hallucination risk.

Ask an ordinary chat question:

```text
你: 你今天看起来很神秘，能给我一点建议吗？
```

Expected behavior: the router should classify this as `persona_chat`, so the NPC continues normal role-play without initializing RAG or forcing unrelated lore.

Inspect retrieval logs:

```bash
tail -n 5 logs/rag_retrieval.jsonl
```

Expected behavior: recent retrieval entries should show the query, whether RAG was used, the top score, and accepted match metadata such as source, category, and score.

Run the test suite:

```bash
python3 -m pytest -v
```

Tests cover deterministic RAG loading, chunking, threshold behavior, logging, and prompt injection behavior. They should not call Claude or require downloading the production embedding model.

---

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

Inventory queries are answered by the local tool layer, for example:

```text
[工具调用: check_inventory] 你有月盐药剂。
```

---

## Available NPCs

| ID | Name | Role |
|----|------|------|
| `blacksmith` | 格林大叔 | 老铁匠，任务偏向探索和收集 |
| `innkeeper` | 罗莎老板娘 | 酒馆老板，任务偏向情报和调查 |
| `mysterious_wizard` | 艾尔文 | 隐居法师，任务偏向魔法遗物和预言 |

---

## Extending the System

To add a new NPC, add an entry to `npc_config.json`:

```json
"new_npc_id": {
  "name": "显示名称",
  "title": "身份头衔",
  "personality": "性格描述",
  "speech_style": "说话风格",
  "background": "背景故事",
  "topics": ["擅长话题"],
  "forbidden_topics": ["禁忌话题"],
  "quest_types": ["任务类型"],
  "greeting": "开场白",
  "farewell": "告别语"
}
```

To add new retrievable world knowledge, place a Markdown file under `knowledge_base/` with useful front matter metadata such as `category` and `npc_id`. Shared lore can use `npc_id: shared`; NPC-specific lore can use the matching NPC ID.

---

## Design Decisions

**Why Claude API instead of a local model or dialogue tree?**

Traditional dialogue trees require manual authoring of every possible branch, which does not scale. Local LLM deployment adds significant infrastructure overhead for a prototype. Using the Claude API allows rapid iteration on prompt design while keeping the focus on narrative quality and system architecture.

**Why JSON-driven NPC configuration?**

Separating character data from logic means designers can add or modify NPCs without touching code. Each NPC is fully defined in `npc_config.json`, including personality, forbidden topics, speech style, and quest type pools.

**Why local RAG?**

RAG keeps world facts outside the prompt template and makes the system easier to expand. Designers can add Markdown lore without changing dialogue code, while intent routing and threshold filtering reduce the risk of injecting irrelevant context into ordinary conversation.

---

## Tech Stack

- Python 3.13
- [Anthropic Claude API](https://www.anthropic.com)
- LangChain
- ChromaDB
- Sentence Transformers / HuggingFace embeddings
- pytest
