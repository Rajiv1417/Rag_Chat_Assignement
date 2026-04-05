"""
Embedder — converts parsed chunks into vector embeddings and stores them in ChromaDB.

Flow:
  ParsedChunk (text/table) → sentence-transformer → ChromaDB
  ParsedChunk (image)      → Gemini VLM summary   → sentence-transformer → ChromaDB
"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.ingestion.parser import ParsedChunk
from src.models.gemini import summarize_image

load_dotenv()

# ── Singleton pattern: load heavy models once ──────────────────────────────────
_embedding_model: SentenceTransformer | None = None
_chroma_client = None
_collection: chromadb.Collection | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(path=db_path)
        _collection = _chroma_client.get_or_create_collection(
            name="tata_motors_docs",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_chunks(chunks: list[ParsedChunk]) -> dict:
    """
    Embed a list of ParsedChunk objects and upsert into ChromaDB.

    For image chunks: calls Gemini VLM first to get a text summary,
    then embeds the summary text.

    Args:
        chunks: Output of parser.parse_pdf()

    Returns:
        Summary dict with counts by chunk type and any errors.
    """
    model = _get_embedding_model()
    collection = _get_collection()

    counts = {"text": 0, "table": 0, "image": 0, "errors": 0}
    error_log: list[str] = []

    for chunk in chunks:
        try:
            text_to_embed = chunk.text

            # VLM summarization for image chunks
            if chunk.chunk_type == "image" and chunk.image_path:
                try:
                    summary = summarize_image(chunk.image_path)
                    text_to_embed = f"[IMAGE DESCRIPTION] {summary}"
                    chunk.text = text_to_embed  # update in-place for storage
                except Exception as e:
                    error_log.append(f"VLM error for {chunk.image_path}: {e}")
                    text_to_embed = f"[IMAGE on page {chunk.page} of {chunk.source}]"
                    chunk.text = text_to_embed

            # Skip empty chunks
            if not text_to_embed.strip():
                continue

            # Generate embedding
            embedding = model.encode(text_to_embed).tolist()

            # Upsert into ChromaDB
            collection.upsert(
                ids=[chunk.chunk_id],
                embeddings=[embedding],
                documents=[text_to_embed],
                metadatas=[{
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_type": chunk.chunk_type,
                }],
            )

            counts[chunk.chunk_type] += 1

        except Exception as e:
            counts["errors"] += 1
            error_log.append(f"Chunk {chunk.chunk_id}: {e}")

    return {
        "text_chunks": counts["text"],
        "table_chunks": counts["table"],
        "image_chunks": counts["image"],
        "errors": counts["errors"],
        "error_details": error_log,
    }


def get_index_stats() -> dict:
    """Return current ChromaDB collection stats."""
    collection = _get_collection()
    count = collection.count()
    return {"total_chunks": count}
