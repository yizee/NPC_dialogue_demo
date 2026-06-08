from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "npc_knowledge"
MANAGED_CATEGORIES = {"world", "npc", "items", "monsters", "quests", "shops"}


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse a small `key: value` front matter subset, not full YAML."""
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
        text = path.read_text(encoding="utf-8")
        front_matter, content = _parse_front_matter(text)
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

    def _new_store(self) -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _build_vector_store(self) -> Chroma:
        documents = load_knowledge_documents(self.knowledge_dir)
        chunks = split_knowledge_documents(documents)
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        store = self._new_store()
        existing = store.get(include=["metadatas"])
        stale_ids = [
            doc_id
            for doc_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", []))
            if metadata
            and (
                metadata.get("managed_by") == COLLECTION_NAME
                or metadata.get("category") in MANAGED_CATEGORIES
            )
            and doc_id not in ids
        ]
        if stale_ids:
            store.delete(ids=stale_ids)

        for chunk in chunks:
            chunk.metadata["managed_by"] = COLLECTION_NAME
        store.add_documents(chunks, ids=ids)
        return store

    def retrieve(self, query: str, npc_id: str) -> RetrievalResult:
        raw_matches = []
        for allowed_npc_id in ("shared", npc_id):
            raw_matches.extend(
                self.vector_store.similarity_search_with_score(
                    query,
                    k=self.k,
                    filter={"npc_id": allowed_npc_id},
                )
            )
        raw_matches.sort(key=lambda item: float(item[1]))

        seen_ids = set()
        matches: list[RetrievalMatch] = []
        for document, score in raw_matches:
            chunk_id = document.metadata.get("chunk_id")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            if document.metadata["npc_id"] not in {"shared", npc_id}:
                continue
            if float(score) > self.distance_threshold:
                continue

            matches.append(
                RetrievalMatch(
                    content=document.page_content,
                    source=document.metadata["source"],
                    category=document.metadata["category"],
                    npc_id=document.metadata["npc_id"],
                    score=float(score),
                )
            )
            if len(matches) >= self.k:
                break

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
