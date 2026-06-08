# LangChain Chroma RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local LangChain + Chroma RAG path that enriches Claude NPC replies with relevant Markdown knowledge while preserving the existing persona chat, `/quest`, and GM feedback flows.

**Architecture:** `rag_pipeline.py` owns document loading, metadata, chunking, Chroma indexing, threshold retrieval, and JSONL logging. `npc_dialogue.py` creates one RAG service per session and asks it for context before each ordinary chat turn; `prompts.py` injects that context without exposing sources to the player. Production uses local HuggingFace embeddings, while tests inject deterministic embeddings and never call Claude or download a model.

**Tech Stack:** Python 3.13, LangChain 1.x, `langchain-community`, `langchain-text-splitters`, `langchain-chroma`, `langchain-huggingface`, ChromaDB, Sentence Transformers, Anthropic SDK, pytest

---

## File Map

- Create `rag_pipeline.py`: focused RAG service and retrieval result types.
- Create `knowledge_base/**/*.md`: world, NPC, item, monster, quest, and shop facts.
- Create `tests/test_rag_pipeline.py`: loader, chunking, threshold, filtering, and logging tests.
- Create `tests/test_prompts.py`: RAG prompt behavior tests.
- Create `requirements.txt`: reproducible runtime and test dependencies.
- Modify `prompts.py`: optional retrieved context in the persona prompt.
- Modify `npc_dialogue.py`: initialize RAG and invoke it on normal dialogue turns.
- Modify `README.md`: setup, architecture, indexing, verification, and troubleshooting.
- Create `.gitignore`: exclude local vector data, logs, caches, and secrets.

> Repository note: `/Users/shiyize/Desktop/GameNPCDialogue` is currently inside the Git repository rooted at `/Users/shiyize`. Do not run the commit steps below until this project has its own Git repository or the user explicitly approves committing through the parent repository.

### Task 1: Create Reproducible Dependencies and Ignore Runtime Artifacts

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Write dependency declarations**

Create `requirements.txt`:

```text
anthropic>=0.84,<1
chromadb>=1.5,<2
langchain>=1.3,<2
langchain-chroma>=1.1,<2
langchain-community>=0.4,<1
langchain-huggingface>=1.2,<2
langchain-text-splitters>=1.1,<2
sentence-transformers>=5.2,<6
pytest>=9,<10
```

- [ ] **Step 2: Ignore generated and secret files**

Create `.gitignore`:

```gitignore
.env
.DS_Store
__pycache__/
*.py[cod]
.pytest_cache/

chroma_db/
logs/
```

- [ ] **Step 3: Install the missing integration and verify imports**

Run:

```bash
python3 -m pip install -r requirements.txt
python3 -c "from langchain_chroma import Chroma; from langchain_huggingface import HuggingFaceEmbeddings; from langchain_text_splitters import RecursiveCharacterTextSplitter; print('RAG dependencies OK')"
```

Expected:

```text
RAG dependencies OK
```

- [ ] **Step 4: Commit when the project has an isolated repository**

```bash
git add requirements.txt .gitignore
git commit -m "build: add RAG dependencies and ignores"
```

### Task 2: Add the Markdown Knowledge Base

**Files:**
- Create: `knowledge_base/world/world_lore.md`
- Create: `knowledge_base/npcs/blacksmith.md`
- Create: `knowledge_base/npcs/innkeeper.md`
- Create: `knowledge_base/npcs/mysterious_wizard.md`
- Create: `knowledge_base/items/items.md`
- Create: `knowledge_base/monsters/monsters.md`
- Create: `knowledge_base/quests/quest_rules.md`
- Create: `knowledge_base/shops/shop_rules.md`

- [ ] **Step 1: Add shared world lore**

Create `knowledge_base/world/world_lore.md`:

```markdown
---
category: world
npc_id: shared
---

# 灰烬谷与月湾镇

月湾镇位于灰烬谷南缘，是北方矿道与沿海商路的交汇点。镇中心的醉月酒馆是旅人交换消息的地方，老铁匠铺负责修理矿工与冒险者的装备。

镇北是霜牙沼泽。三十年前发生的星落事件让沼泽残留不稳定的魔法波动，也产生了被称为碎星髓的稀有矿物。

镇外东侧矗立着艾尔文的旧塔。居民知道塔中住着一位老法师，但很少有人被允许进入。
```

- [ ] **Step 2: Add NPC-specific knowledge**

Create `knowledge_base/npcs/blacksmith.md`:

```markdown
---
category: npc
npc_id: blacksmith
---

# 格林大叔

格林在月湾镇打铁三十年，熟悉矿石、武器保养和灰烬谷常见怪物的弱点。他的儿子莱恩五年前随北方巡逻队失踪，最后的线索指向霜牙沼泽。

格林不懂高深魔法理论。遇到魔法问题时，他通常会建议玩家去找艾尔文。
```

Create `knowledge_base/npcs/innkeeper.md`:

```markdown
---
category: npc
npc_id: innkeeper
---

# 罗莎老板娘

罗莎经营醉月酒馆二十年，掌握旅客、商队和镇上异常事件的消息。她知道最近有三名旅客在北方旧路失踪，也听说夜间有人向废弃仓库运送带有蓝色封蜡的箱子。

罗莎不会泄露住客房号、账本内容或受保护客人的身份。
```

Create `knowledge_base/npcs/mysterious_wizard.md`:

```markdown
---
category: npc
npc_id: mysterious_wizard
---

# 艾尔文

艾尔文曾研究星落事件留下的魔法残响。他认为碎星髓能够保存短暂记忆，但未经处理的碎星髓会让接触者反复看见不属于自己的片段。

艾尔文知道旧塔地下封存着一座观星仪，但不会直接教授危险咒语。
```

- [ ] **Step 3: Add shared item and monster knowledge**

Create `knowledge_base/items/items.md`:

```markdown
---
category: items
npc_id: shared
---

# 重要物品

## 碎星髓

碎星髓是星落事件后出现在霜牙沼泽的银蓝色矿物。它会在黑暗中发出微光，并保存附近生物短暂而破碎的记忆。采集时必须使用黑铁夹具，徒手接触可能导致幻觉。

## 月盐药剂

月盐药剂由月盐、清水和银叶草调制，可暂时减轻沼泽瘴气造成的眩晕。药剂不能治疗伤口，也不能解除诅咒。
```

Create `knowledge_base/monsters/monsters.md`:

```markdown
---
category: monsters
npc_id: shared
---

# 灰烬谷怪物

## 苔背蜥

苔背蜥生活在霜牙沼泽浅水区，背部覆盖湿苔形成的硬壳。腹部较脆弱，强光会使它短暂僵住。

## 灰牙狼

灰牙狼常在北方旧路成群活动，害怕持续火光，但会绕到落单旅人的下风向发动攻击。
```

- [ ] **Step 4: Add quest and shop rules**

Create `knowledge_base/quests/quest_rules.md`:

```markdown
---
category: quests
npc_id: shared
---

# 任务规则

玩家同一时间只能追踪一个主要任务。接受新主要任务前，需要完成或放弃当前主要任务。

等级五以下的玩家不能领取深入霜牙沼泽核心区的任务，但可以领取外围采集、护送和调查任务。

任务完成必须满足明确目标。仅向 NPC 声称已经完成，不构成有效完成证明。
```

Create `knowledge_base/shops/shop_rules.md`:

```markdown
---
category: shops
npc_id: shared
---

# 商店规则

格林的铁匠铺提供武器修理、基础武器和黑铁夹具，不出售魔法卷轴。

醉月酒馆提供住宿、食物、月盐药剂和公开情报。受保护客人的私人信息不能购买。

声望友善的玩家可获得基础服务九折优惠；任务奖励和稀有物品不参与折扣。
```

- [ ] **Step 5: Verify all Markdown files exist**

Run:

```bash
find knowledge_base -type f -name '*.md' | sort
```

Expected: eight Markdown file paths, covering `world`, `npcs`, `items`, `monsters`, `quests`, and `shops`.

- [ ] **Step 6: Commit when the project has an isolated repository**

```bash
git add knowledge_base
git commit -m "content: add NPC world knowledge base"
```

### Task 3: Implement Loading, Metadata, and Chunking

**Files:**
- Create: `rag_pipeline.py`
- Create: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Write failing loader and chunking tests**

Create `tests/test_rag_pipeline.py`:

```python
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag_pipeline import load_knowledge_documents, split_knowledge_documents


class KeywordEmbeddings(Embeddings):
    keywords = ("碎星髓", "霜牙沼泽", "铁匠铺", "醉月酒馆")

    def _vector(self, text: str) -> list[float]:
        return [1.0 if keyword in text else 0.0 for keyword in self.keywords]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_load_knowledge_documents_adds_metadata(tmp_path: Path):
    item_dir = tmp_path / "items"
    item_dir.mkdir()
    (item_dir / "items.md").write_text(
        "---\ncategory: items\nnpc_id: shared\n---\n\n# 碎星髓\n\n来自霜牙沼泽。",
        encoding="utf-8",
    )

    documents = load_knowledge_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["category"] == "items"
    assert documents[0].metadata["npc_id"] == "shared"
    assert documents[0].metadata["source"].endswith("items/items.md")
    assert "碎星髓" in documents[0].page_content


def test_split_knowledge_documents_preserves_metadata(tmp_path: Path):
    world_dir = tmp_path / "world"
    world_dir.mkdir()
    (world_dir / "world_lore.md").write_text(
        "---\ncategory: world\nnpc_id: shared\n---\n\n" + "月湾镇。" * 80,
        encoding="utf-8",
    )

    documents = load_knowledge_documents(tmp_path)
    chunks = split_knowledge_documents(documents, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["category"] == "world" for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
python3 -m pytest tests/test_rag_pipeline.py -v
```

Expected: collection failure with `ModuleNotFoundError: No module named 'rag_pipeline'`.

- [ ] **Step 3: Implement document loading and chunking**

Create `rag_pipeline.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    end_index = text.find("\n---\n", 4)
    if end_index == -1:
        return {}, text

    metadata: dict[str, str] = {}
    for line in text[4:end_index].splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip()
    return metadata, text[end_index + 5 :].lstrip()


def load_knowledge_documents(knowledge_dir: str | Path) -> list[Document]:
    root = Path(knowledge_dir)
    documents: list[Document] = []

    for path in sorted(root.rglob("*.md")):
        loaded = TextLoader(str(path), encoding="utf-8").load()[0]
        front_matter, content = _parse_front_matter(loaded.page_content)
        relative_source = path.relative_to(root).as_posix()
        metadata = {
            "source": relative_source,
            "category": front_matter.get("category", path.parent.name),
            "npc_id": front_matter.get("npc_id", "shared"),
        }
        documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise ValueError(f"No Markdown knowledge files found in {root}")
    return documents


def split_knowledge_documents(
    documents: Sequence[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}::{index}"
    return chunks
```

- [ ] **Step 4: Run loader tests**

Run:

```bash
python3 -m pytest tests/test_rag_pipeline.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit when the project has an isolated repository**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "feat: load and chunk NPC knowledge"
```

### Task 4: Implement Chroma Indexing and Threshold Retrieval

**Files:**
- Modify: `rag_pipeline.py`
- Modify: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Add failing retrieval tests**

Append to `tests/test_rag_pipeline.py`:

```python
from rag_pipeline import RagService


def _write_test_knowledge(root: Path) -> None:
    item_dir = root / "items"
    npc_dir = root / "npcs"
    item_dir.mkdir(parents=True)
    npc_dir.mkdir(parents=True)
    (item_dir / "items.md").write_text(
        "---\ncategory: items\nnpc_id: shared\n---\n\n碎星髓来自霜牙沼泽。",
        encoding="utf-8",
    )
    (npc_dir / "wizard.md").write_text(
        "---\ncategory: npc\nnpc_id: mysterious_wizard\n---\n\n铁匠铺不知道的观星仪秘密。",
        encoding="utf-8",
    )


def test_retrieve_returns_relevant_shared_knowledge(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    _write_test_knowledge(knowledge_dir)
    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        log_path=tmp_path / "rag.jsonl",
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
    )

    result = service.retrieve("碎星髓在哪里？", npc_id="blacksmith")

    assert result.used_rag is True
    assert "霜牙沼泽" in result.context
    assert result.matches[0].category == "items"


def test_retrieve_rejects_irrelevant_and_private_knowledge(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    _write_test_knowledge(knowledge_dir)
    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        log_path=tmp_path / "rag.jsonl",
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
    )

    irrelevant = service.retrieve("今天天气好吗？", npc_id="blacksmith")
    private = service.retrieve("观星仪秘密", npc_id="blacksmith")

    assert irrelevant.used_rag is False
    assert irrelevant.context == ""
    assert private.used_rag is False
```

- [ ] **Step 2: Run the retrieval tests and confirm failure**

Run:

```bash
python3 -m pytest tests/test_rag_pipeline.py -v
```

Expected: import failure because `RagService` is not yet defined.

- [ ] **Step 3: Add result types and the RAG service**

Append to `rag_pipeline.py`:

```python
@dataclass(frozen=True)
class RetrievalMatch:
    content: str
    source: str
    category: str
    npc_id: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    used_rag: bool
    context: str
    matches: tuple[RetrievalMatch, ...]

    @property
    def top_score(self) -> float | None:
        return self.matches[0].score if self.matches else None


class RagService:
    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge_base",
        persist_dir: str | Path = "chroma_db",
        log_path: str | Path = "logs/rag_retrieval.jsonl",
        embeddings: Embeddings | None = None,
        distance_threshold: float = 0.7,
        k: int = 4,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.persist_dir = Path(persist_dir)
        self.log_path = Path(log_path)
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=DEFAULT_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.distance_threshold = distance_threshold
        self.k = k
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self) -> Chroma:
        documents = load_knowledge_documents(self.knowledge_dir)
        chunks = split_knowledge_documents(documents)
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        store = Chroma(
            collection_name="npc_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        store.delete(ids=ids)
        store.add_documents(chunks, ids=ids)
        return store

    def retrieve(self, query: str, npc_id: str) -> RetrievalResult:
        raw_matches = self.vector_store.similarity_search_with_score(
            query,
            k=max(self.k * 3, self.k),
        )
        matches = [
            RetrievalMatch(
                content=document.page_content,
                source=document.metadata["source"],
                category=document.metadata["category"],
                npc_id=document.metadata["npc_id"],
                score=float(score),
            )
            for document, score in raw_matches
            if document.metadata["npc_id"] in {"shared", npc_id}
            and float(score) <= self.distance_threshold
        ][: self.k]
        context = "\n\n".join(match.content for match in matches)
        result = RetrievalResult(
            query=query,
            used_rag=bool(matches),
            context=context,
            matches=tuple(matches),
        )
        self.log_retrieval(npc_id, result)
        return result

    def log_retrieval(self, npc_id: str, result: RetrievalResult) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "npc_id": npc_id,
            "query": result.query,
            "used_rag": result.used_rag,
            "top_score": result.top_score,
            "matches": [
                {
                    "source": match.source,
                    "category": match.category,
                    "score": match.score,
                }
                for match in result.matches
            ],
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run retrieval tests**

Run:

```bash
python3 -m pytest tests/test_rag_pipeline.py -v
```

Expected: all four tests pass. If the deterministic test produces Chroma distance values different from the initial expectation, inspect the returned scores and adjust only the test threshold, not the filtering logic.

- [ ] **Step 5: Commit when the project has an isolated repository**

```bash
git add rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "feat: add Chroma threshold retrieval"
```

### Task 5: Verify JSONL Retrieval Logging

**Files:**
- Modify: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Add a logging test**

Append to `tests/test_rag_pipeline.py`:

```python
import json


def test_retrieve_writes_valid_jsonl_log(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    _write_test_knowledge(knowledge_dir)
    log_path = tmp_path / "logs" / "rag.jsonl"
    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        log_path=log_path,
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
    )

    service.retrieve("碎星髓在哪里？", npc_id="blacksmith")

    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["npc_id"] == "blacksmith"
    assert record["used_rag"] is True
    assert record["matches"][0]["source"] == "items/items.md"
    assert "api" not in json.dumps(record).lower()
```

- [ ] **Step 2: Run the logging test**

Run:

```bash
python3 -m pytest tests/test_rag_pipeline.py::test_retrieve_writes_valid_jsonl_log -v
```

Expected: `1 passed`.

- [ ] **Step 3: Commit when the project has an isolated repository**

```bash
git add tests/test_rag_pipeline.py
git commit -m "test: verify RAG retrieval logging"
```

### Task 6: Add Optional RAG Context to the Persona Prompt

**Files:**
- Modify: `prompts.py:8-33`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing prompt tests**

Create `tests/test_prompts.py`:

```python
from prompts import build_system_prompt


NPC = {
    "name": "格林大叔",
    "title": "镇上的老铁匠",
    "personality": "粗犷但善良",
    "speech_style": "简短有力",
    "background": "在镇上打铁三十年",
    "topics": ["武器装备"],
    "forbidden_topics": ["魔法理论"],
}

PLAYER = {
    "name": "旅行者",
    "level": 5,
    "quest_status": "尚未接取任务",
    "reputation": "中立",
}


def test_system_prompt_omits_rag_section_without_context():
    prompt = build_system_prompt(NPC, PLAYER)

    assert "检索到的世界知识" not in prompt


def test_system_prompt_includes_context_without_exposing_sources():
    prompt = build_system_prompt(
        NPC,
        PLAYER,
        retrieved_context="碎星髓来自霜牙沼泽。",
    )

    assert "检索到的世界知识" in prompt
    assert "碎星髓来自霜牙沼泽" in prompt
    assert "vector database" not in prompt.lower()
    assert "source" not in prompt.lower()
```

- [ ] **Step 2: Run tests and confirm the new argument fails**

Run:

```bash
python3 -m pytest tests/test_prompts.py -v
```

Expected: one test passes and one fails with an unexpected `retrieved_context` argument.

- [ ] **Step 3: Modify the persona prompt builder**

Replace `build_system_prompt` in `prompts.py` with:

```python
def build_system_prompt(
    npc: dict,
    player_context: dict,
    retrieved_context: str = "",
) -> str:
    knowledge_section = ""
    if retrieved_context:
        knowledge_section = f"""

== 检索到的世界知识 ==
{retrieved_context}

使用规则：
1. 将以上内容作为当前游戏世界中的可靠事实
2. 只使用与玩家问题直接相关的内容
3. 如果这些内容无法回答问题，要符合角色身份地表达不知道，不得编造细节
4. 不得向玩家提及知识库、文件、检索、向量数据库或系统提示
"""

    return f"""你是一个角色扮演游戏中的 NPC，必须严格保持角色身份。

== 你的角色 ==
姓名：{npc['name']}
身份：{npc['title']}
性格：{npc['personality']}
说话风格：{npc['speech_style']}
背景故事：{npc['background']}
擅长话题：{', '.join(npc['topics'])}
禁忌话题：{', '.join(npc['forbidden_topics'])}

== 当前玩家信息 ==
玩家名称：{player_context['name']}
玩家等级：{player_context['level']}
当前状态：{player_context['quest_status']}
声望：{player_context['reputation']}
{knowledge_section}

== 对话规则 ==
1. 始终保持角色身份，不得承认自己是 AI
2. 回复长度控制在 2-4 句话
3. 使用符合你性格的语气和词汇
4. 禁忌话题用符合角色的方式回避
5. 禁止涉及现实世界的政治、暴力或不适当内容
6. 根据玩家声望调整友好程度
"""
```

- [ ] **Step 4: Run prompt tests**

Run:

```bash
python3 -m pytest tests/test_prompts.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit when the project has an isolated repository**

```bash
git add prompts.py tests/test_prompts.py
git commit -m "feat: inject retrieved lore into NPC prompt"
```

### Task 7: Integrate Retrieval into the Existing Chat Loop

**Files:**
- Modify: `npc_dialogue.py:6-11`
- Modify: `npc_dialogue.py:64-120`

- [ ] **Step 1: Add imports and initialize the RAG service lazily**

Change the imports in `npc_dialogue.py` to:

```python
import json
import os

from anthropic import Anthropic

from prompts import build_gm_prompt, build_quest_prompt, build_system_prompt
from rag_pipeline import RagService
```

At the start of `chat_with_npc`, replace the static system prompt initialization with:

```python
def chat_with_npc(npc_id: str, player_context: dict):
    npc = load_npc_config(npc_id)
    conversation_history = []

    try:
        rag_service = RagService()
    except Exception as exc:
        rag_service = None
        print(f"[RAG 警告] 知识库初始化失败，将继续使用普通 NPC 对话：{exc}\n")
```

This keeps the existing chat usable if the local embedding model or Chroma initialization fails.

- [ ] **Step 2: Retrieve before each normal dialogue turn**

Immediately before appending the normal user message, replace the existing static-prompt behavior with:

```python
        retrieved_context = ""
        if rag_service is not None:
            try:
                retrieval = rag_service.retrieve(player_input, npc_id=npc_id)
                retrieved_context = retrieval.context
            except Exception as exc:
                print(f"[RAG 警告] 本轮检索失败，将使用普通对话：{exc}\n")

        system_prompt = build_system_prompt(
            npc,
            player_context,
            retrieved_context=retrieved_context,
        )
        conversation_history.append({"role": "user", "content": player_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=conversation_history,
        )
```

Do not change the `/quest` or `quit` branches in this phase.

- [ ] **Step 3: Run syntax and unit checks**

Run:

```bash
python3 -m py_compile npc_dialogue.py prompts.py rag_pipeline.py
python3 -m pytest -v
```

Expected: compilation succeeds and all tests pass.

- [ ] **Step 4: Verify startup without making a Claude request**

Run:

```bash
printf 'blacksmith\nquit\n' | python3 npc_dialogue.py
```

Expected:
- The CLI lists the NPC and greeting.
- RAG initializes or prints a clear fallback warning.
- The program exits without calling GM feedback because there is no conversation history.

- [ ] **Step 5: Commit when the project has an isolated repository**

```bash
git add npc_dialogue.py
git commit -m "feat: route NPC chat through local RAG"
```

### Task 8: Document Setup and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update overview and architecture**

Add a RAG module to the overview:

```markdown
**Local RAG Knowledge System** — Every normal dialogue turn is searched against a Markdown knowledge base using local multilingual embeddings and Chroma. Relevant lore is injected into Claude's system prompt, while irrelevant social chat continues through the original persona path.
```

Add this architecture section:

```markdown
## RAG Architecture

```text
Player input
  -> LangChain document retriever
  -> local multilingual embedding
  -> Chroma similarity search
  -> relevance threshold
     -> relevant: inject lore into Claude persona prompt
     -> irrelevant: use persona prompt without RAG context
  -> log retrieval metadata to logs/rag_retrieval.jsonl
```

Sources remain hidden during gameplay to preserve immersion. Retrieval evidence is stored in the backend log for debugging and demonstration.
```

- [ ] **Step 2: Replace installation instructions**

Use:

```markdown
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
export ANTHROPIC_API_KEY="your-key"
```

**4. Run**

```bash
python3 npc_dialogue.py
```

The embedding model is downloaded on the first run. Chroma data is stored locally in `chroma_db/`.
```

- [ ] **Step 3: Add verification examples**

Add:

```markdown
## RAG Verification

Try a knowledge question:

```text
你: 碎星髓是什么？采集时需要注意什么？
```

Expected behavior: the NPC explains that it is a silver-blue mineral from the starfall event and mentions black-iron tools or hallucination risk.

Try ordinary chat:

```text
你: 今天过得怎么样？
```

Expected behavior: the NPC replies in character without presenting unrelated lore or source citations.

Inspect backend retrieval evidence:

```bash
tail -n 5 logs/rag_retrieval.jsonl
```

Run automated tests:

```bash
python3 -m pytest -v
```
```

- [ ] **Step 4: Update project structure**

Include:

```text
├── knowledge_base/       # Markdown lore and game rules
├── tests/                # Offline RAG and prompt tests
├── rag_pipeline.py       # Loading, chunking, Chroma retrieval, logging
├── npc_dialogue.py       # Claude CLI and RAG orchestration
├── prompts.py            # Persona, RAG, quest, and GM prompts
├── npc_config.json       # NPC definitions
├── requirements.txt      # Reproducible dependencies
└── README.md
```

- [ ] **Step 5: Commit when the project has an isolated repository**

```bash
git add README.md
git commit -m "docs: explain LangChain Chroma RAG"
```

### Task 9: Final Verification and Threshold Calibration

**Files:**
- Modify if calibration requires it: `rag_pipeline.py`
- Modify if behavior changed: `README.md`

- [ ] **Step 1: Run the complete offline verification suite**

Run:

```bash
python3 -m py_compile npc_dialogue.py prompts.py rag_pipeline.py
python3 -m pytest -v
```

Expected: compilation succeeds and all tests pass without an Anthropic API call.

- [ ] **Step 2: Inspect real retrieval scores**

Run:

```bash
python3 - <<'PY'
from rag_pipeline import RagService

service = RagService()
for query in [
    "碎星髓是什么？",
    "霜牙沼泽在哪里？",
    "铁匠铺卖魔法卷轴吗？",
    "你好，今天怎么样？",
]:
    result = service.retrieve(query, npc_id="blacksmith")
    print(query, result.used_rag, result.top_score, [m.source for m in result.matches])
PY
```

Expected:
- The first three queries retrieve the relevant item, world, or shop documents.
- The social query should ideally be above the accepted distance threshold and return `used_rag=False`.

- [ ] **Step 3: Calibrate one explicit threshold**

If the real model's score distribution does not separate the examples, change only the default `distance_threshold` in `RagService.__init__`. Choose the lowest value that accepts all three knowledge queries while rejecting the social query, then rerun Step 2 and record the final value in the README.

- [ ] **Step 4: Run one live Claude smoke test**

Run:

```bash
python3 npc_dialogue.py
```

Manual sequence:

```text
blacksmith
碎星髓是什么？采集时需要注意什么？
今天过得怎么样？
quit
```

Expected:
- The first reply uses knowledge-base facts while remaining in character.
- The second reply does not mention unrelated retrieved lore.
- No source path appears in player-facing output.
- `logs/rag_retrieval.jsonl` contains both turns.
- `quit` still produces the existing GM report.

- [ ] **Step 5: Inspect the working tree**

Run:

```bash
git status --short
```

Expected: only intended GameNPCDialogue files are listed. Because the current Git root is `/Users/shiyize`, do not stage or commit from that parent repository until repository isolation is resolved.
