import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline import RagService, load_knowledge_documents, split_knowledge_documents


class KeywordEmbeddings(Embeddings):
    keywords = ("碎星髓", "霜牙沼泽", "铁匠铺", "醉月酒馆", "观星仪", "DELETE_ME")

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
    wizard_private = service.retrieve("观星仪秘密", npc_id="mysterious_wizard")

    assert irrelevant.used_rag is False
    assert irrelevant.context == ""
    assert private.used_rag is False
    assert wizard_private.used_rag is True
    assert "观星仪秘密" in wizard_private.context


def test_initialization_preserves_unrelated_persist_dir_files(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    _write_test_knowledge(knowledge_dir)
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    unrelated_file = persist_dir / "do_not_delete.txt"
    unrelated_file.write_text("keep me", encoding="utf-8")

    RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=persist_dir,
        log_path=tmp_path / "rag.jsonl",
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
    )

    assert unrelated_file.read_text(encoding="utf-8") == "keep me"


def test_initialization_preserves_unrelated_collection_ids(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    _write_test_knowledge(knowledge_dir)
    persist_dir = tmp_path / "chroma"
    embeddings = KeywordEmbeddings()
    existing_store = Chroma(
        collection_name="npc_knowledge",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    existing_store.add_documents(
        [
            Document(
                page_content="外部资料，不属于本次知识库加载。",
                metadata={
                    "source": "external.md",
                    "category": "external",
                    "npc_id": "shared",
                    "chunk_id": "external.md::0",
                },
            )
        ],
        ids=["external.md::0"],
    )

    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=persist_dir,
        log_path=tmp_path / "rag.jsonl",
        embeddings=embeddings,
        distance_threshold=0.4,
    )

    existing = service.vector_store.get(ids=["external.md::0"], include=["documents"])
    assert existing["ids"] == ["external.md::0"]
    assert existing["documents"] == ["外部资料，不属于本次知识库加载。"]


def test_initialization_prunes_deleted_managed_markdown_chunks(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    items_dir = knowledge_dir / "items"
    items_dir.mkdir(parents=True)
    stale_file = items_dir / "stale.md"
    stale_file.write_text(
        "---\ncategory: items\nnpc_id: shared\n---\n\nDELETE_ME 这条旧设定会被删除。",
        encoding="utf-8",
    )
    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        log_path=tmp_path / "rag.jsonl",
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
    )
    assert service.retrieve("DELETE_ME", npc_id="blacksmith").used_rag is True

    stale_file.unlink()
    (items_dir / "items.md").write_text(
        "---\ncategory: items\nnpc_id: shared\n---\n\n碎星髓来自霜牙沼泽。",
        encoding="utf-8",
    )
    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        log_path=tmp_path / "rag.jsonl",
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
    )

    assert service.retrieve("DELETE_ME", npc_id="blacksmith").used_rag is False


def test_retrieve_uses_metadata_filter_before_private_docs_can_crowd_out_shared(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    item_dir = knowledge_dir / "items"
    npc_dir = knowledge_dir / "npcs"
    item_dir.mkdir(parents=True)
    npc_dir.mkdir(parents=True)
    (item_dir / "items.md").write_text(
        "---\ncategory: items\nnpc_id: shared\n---\n\n碎星髓是所有 NPC 都可以引用的公开设定。",
        encoding="utf-8",
    )
    for index in range(6):
        (npc_dir / f"private_{index}.md").write_text(
            f"---\ncategory: npc\nnpc_id: mysterious_wizard\n---\n\n碎星髓 私有观测记录 {index}。",
            encoding="utf-8",
        )

    service = RagService(
        knowledge_dir=knowledge_dir,
        persist_dir=tmp_path / "chroma",
        log_path=tmp_path / "rag.jsonl",
        embeddings=KeywordEmbeddings(),
        distance_threshold=0.4,
        k=1,
    )

    result = service.retrieve("碎星髓", npc_id="blacksmith")

    assert result.used_rag is True
    assert result.matches[0].npc_id == "shared"
    assert "公开设定" in result.context


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
