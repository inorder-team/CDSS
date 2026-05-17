"""
CDSS Platform – RAG Engine (ChromaDB + Sentence Transformers)
Ingests clinical guidelines into ChromaDB and performs semantic retrieval
for evidence-based decision support.
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.models.schemas import EvidenceDocument

settings = get_settings()


# ─────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────

def chunk_document(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Split document into overlapping chunks by sentence boundary."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap
            words = current.split()
            current = " ".join(words[-overlap // 6 :]) + " " + sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 50]


# ─────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────

class CDSSRagEngine:
    """
    Manages ChromaDB collection for clinical guidelines.
    Uses local sentence-transformers embeddings (no external calls).
    """

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedder: Optional[SentenceTransformer] = None
        self._ready = False

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(f"[RAG] Loading embedding model: {settings.embedding_model}")
            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.chroma_collection_clinical,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def is_ready(self) -> bool:
        return self._ready

    def ingest_guidelines_directory(self, directory: str | Path) -> int:
        """
        Ingest all .txt guideline files in the given directory into ChromaDB.
        Returns count of chunks ingested.
        """
        directory = Path(directory)
        total_chunks = 0
        collection = self._get_collection()
        embedder = self._get_embedder()

        for filepath in directory.glob("*.txt"):
            logger.info(f"[RAG] Ingesting: {filepath.name}")
            text = filepath.read_text(encoding="utf-8")

            # Extract document metadata from header
            doc_id = filepath.stem
            for line in text.splitlines()[:5]:
                if line.startswith("DOCUMENT_ID:"):
                    doc_id = line.split(":", 1)[1].strip()
                    break

            chunks = chunk_document(text)
            logger.info(f"[RAG] {filepath.name} → {len(chunks)} chunks")

            # Check which chunks already exist
            existing_ids = set()
            try:
                existing = collection.get(where={"source_doc": doc_id})
                existing_ids = set(existing["ids"])
            except Exception:
                pass

            new_chunks = []
            new_ids = []
            new_meta = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i:04d}"
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
                    new_ids.append(chunk_id)
                    new_meta.append({
                        "source_doc": doc_id,
                        "chunk_index": i,
                        "filename": filepath.name,
                        "doc_type": "clinical_guideline",
                    })

            if new_chunks:
                embeddings = embedder.encode(new_chunks, show_progress_bar=False).tolist()
                collection.add(
                    documents=new_chunks,
                    embeddings=embeddings,
                    ids=new_ids,
                    metadatas=new_meta,
                )
                total_chunks += len(new_chunks)
                logger.info(f"[RAG] Added {len(new_chunks)} new chunks from {filepath.name}")
            else:
                logger.info(f"[RAG] All chunks already in collection for {filepath.name}")
                total_chunks += len(chunks)

        self._ready = True
        logger.info(f"[RAG] Ingestion complete. Total chunks: {total_chunks}")
        return total_chunks

    def retrieve(
        self,
        query: str,
        patient_context_summary: str = "",
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[EvidenceDocument]:
        """
        Perform semantic similarity retrieval against clinical guidelines.
        Returns ranked list of EvidenceDocument.
        """
        top_k = top_k or settings.rag_top_k
        score_threshold = score_threshold or settings.rag_similarity_threshold

        collection = self._get_collection()
        embedder = self._get_embedder()

        # Guard: if collection is empty, querying will raise an exception
        doc_count = collection.count()
        if doc_count == 0:
            logger.warning("[RAG] Collection is empty — no guidelines ingested yet")
            return []

        # Enrich query with patient context summary for better retrieval
        full_query = f"{query} {patient_context_summary}".strip()
        query_embedding = embedder.encode([full_query], show_progress_bar=False).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, doc_count),
            include=["documents", "metadatas", "distances"],
        )

        evidence: list[EvidenceDocument] = []
        if not results["documents"] or not results["documents"][0]:
            return evidence

        for doc_text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity score
            similarity = max(0.0, 1.0 - distance)
            if similarity < score_threshold:
                continue

            evidence.append(
                EvidenceDocument(
                    doc_id=meta.get("source_doc", "unknown"),
                    source=meta.get("filename", "guideline"),
                    content_snippet=doc_text[:500],
                    similarity_score=round(similarity, 4),
                    source_type="clinical_guideline",
                    relevance_tag=_classify_chunk(doc_text),
                )
            )

        # Sort by similarity descending
        evidence.sort(key=lambda e: e.similarity_score, reverse=True)
        return evidence

    def collection_count(self) -> int:
        try:
            count = self._get_collection().count()
            # If collection already has data (e.g. persisted from a previous run),
            # mark the engine as ready without needing to re-ingest
            if count > 0:
                self._ready = True
            return count
        except Exception:
            return 0


def _classify_chunk(text: str) -> str:
    """Naive keyword-based tag for evidence chunk relevance."""
    text_lower = text.lower()
    if "aspirin" in text_lower and "allergy" in text_lower:
        return "antiplatelet_allergy"
    if "antiplatelet" in text_lower or "dapt" in text_lower:
        return "antiplatelet"
    if "risk stratif" in text_lower or "grace" in text_lower:
        return "risk_stratification"
    if "invasive" in text_lower or "angiograph" in text_lower or "pci" in text_lower:
        return "invasive_strategy"
    if "renal" in text_lower or "egfr" in text_lower or "ckd" in text_lower:
        return "renal"
    if "human review" in text_lower or "cardiologist review" in text_lower:
        return "ai_safety"
    if "beta-block" in text_lower or "statin" in text_lower or "ace" in text_lower:
        return "adjunct_therapy"
    if "monitor" in text_lower:
        return "monitoring"
    return "general"


# Singleton
rag_engine = CDSSRagEngine()
