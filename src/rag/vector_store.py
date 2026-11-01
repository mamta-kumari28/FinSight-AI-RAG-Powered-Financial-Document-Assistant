"""Content-addressed, per-browser-session document indexes. No default model downloads."""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from filelock import FileLock
from langchain_core.documents import Document

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.errors import FinSightError
from src.ingestion import load_document
from src.ingestion.text_splitter import split_documents


class Embedder(Protocol):
    signature: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class IndexRecord:
    id: str
    filename: str
    chunks: int
    content_hash: str
    signature: str


@dataclass(frozen=True)
class Evidence:
    text: str
    metadata: dict
    distance: float

    @property
    def source(self) -> str:
        m = self.metadata
        if "page" in m:
            return f"{m['filename']} · page {m['page']}"
        return f"{m['filename']} · sheet {m['sheet']} · row {m['row_start']}"


class DocumentStore:
    def __init__(self, root: Path, session_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise FinSightError("Invalid local document session. Reset the session.")
        self.path = Path(root) / session_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False))
        self.lock = FileLock(str(self.path / "index.lock"), timeout=60)

    def list_indexes(self) -> list[IndexRecord]:
        result = []
        for collection in self.client.list_collections():
            metadata = collection.metadata or {}
            if metadata.get("state") == "ready":
                result.append(IndexRecord(collection.name, metadata["filename"], collection.count(), metadata["content_hash"], metadata["signature"]))
        return sorted(result, key=lambda record: record.filename)

    def index(self, data: bytes, filename: str, embedder: Embedder) -> tuple[IndexRecord, bool]:
        # Parsing/validation happens even for a cache hit; bytes are never stored as uploads.
        documents = load_document(data, filename)
        filename = documents[0].metadata["filename"]
        content_hash = hashlib.sha256(data).hexdigest()
        spec = json.dumps([content_hash, filename, embedder.signature, CHUNK_SIZE, CHUNK_OVERLAP, "schema-1"])
        name = "doc-" + hashlib.sha256(spec.encode()).hexdigest()[:48]
        with self.lock:
            existing = {collection.name: collection for collection in self.client.list_collections()}
            if name in existing:
                collection = existing[name]
                metadata = collection.metadata or {}
                if metadata.get("state") != "ready" or metadata.get("dimensions") != embedder.dimensions or collection.count() != metadata.get("chunks"):
                    raise FinSightError("A stale or incomplete index exists. Reset this session and index again.")
                return IndexRecord(name, filename, collection.count(), content_hash, embedder.signature), False
            chunks = split_documents(documents)
            if not chunks:
                raise FinSightError("No text chunks are available to index.")
            collection = self.client.create_collection(name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}, metadata={
                "state": "building", "filename": filename,
                "content_hash": content_hash, "signature": embedder.signature,
                "dimensions": embedder.dimensions, "chunks": len(chunks),
            })
            try:
                for offset in range(0, len(chunks), 32):
                    batch = chunks[offset:offset + 32]
                    texts = [chunk.page_content for chunk in batch]
                    vectors = embedder.embed_documents(texts)
                    self._check_vectors(vectors, len(batch), embedder.dimensions)
                    collection.add(ids=[f"chunk-{offset + i}" for i in range(len(batch))],
                                   documents=texts, metadatas=[chunk.metadata for chunk in batch], embeddings=vectors)
                collection.modify(metadata={**collection.metadata, "state": "ready"})
            except Exception:
                self.client.delete_collection(name)
                raise
        return IndexRecord(name, filename, len(chunks), content_hash, embedder.signature), True

    @staticmethod
    def _check_vectors(vectors, count: int, dimensions: int):
        values = np.asarray(vectors, dtype=float)
        if values.shape != (count, dimensions) or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) <= 0).any():
            raise FinSightError("Invalid embedding values or dimension mismatch. Reset the index and check embedding configuration.")

    def retrieve(self, index_ids: list[str], question: str, embedder: Embedder, k: int = 4, max_distance: float = 0.8) -> list[Evidence]:
        if not index_ids or not question.strip() or not 1 <= k <= 12:
            raise FinSightError("Choose indexed documents and enter a question (retrieval count 1–12).")
        with self.lock:
            available = {record.id: record for record in self.list_indexes()}
            if any(name not in available for name in index_ids):
                raise FinSightError("An index was removed or became stale. Select the documents again.")
            if any(available[name].signature != embedder.signature for name in index_ids):
                raise FinSightError("Embedding configuration changed. Remove the old indexes and index your documents again.")
            vector = embedder.embed_query(question)
            self._check_vectors([vector], 1, embedder.dimensions)
            result = []
            for name in index_ids:
                collection = self.client.get_collection(name, embedding_function=None)
                if collection.metadata.get("dimensions") != embedder.dimensions:
                    raise FinSightError("Embedding dimension mismatch. Remove the old index and index again.")
                response = collection.query(query_embeddings=[vector], n_results=min(k, collection.count()), include=["documents", "metadatas", "distances"])
                for text, metadata, distance in zip(response["documents"][0], response["metadatas"][0], response["distances"][0]):
                    if np.isfinite(distance) and distance <= max_distance:
                        result.append(Evidence(text, metadata, float(distance)))
            return sorted(result, key=lambda evidence: evidence.distance)[:k]

    def remove(self, index_id: str):
        with self.lock:
            if index_id not in {collection.name for collection in self.client.list_collections()}:
                raise FinSightError("This document index no longer exists.")
            self.client.delete_collection(index_id)

    def reset(self):
        with self.lock:
            for collection in self.client.list_collections():
                self.client.delete_collection(collection.name)
