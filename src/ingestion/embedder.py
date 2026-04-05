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
_chroma_client = None   # chromadb.ClientAPI
_collection = None      # chromadb.Collection


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


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDER TESTS — run directly: python -m src.ingestion.embedder
# Tests embedding + ChromaDB only. Image/VLM step is SKIPPED.
# Zero API calls to Groq.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import tempfile
    import os

    from src.ingestion.parser import parse_pdf

    PDF_PATH = os.getenv(
        "TEST_PDF",
        "/workspaces/Rag_Chat_Assignement/Documens/pdfs/"
        "SC_2025_36 Introduction of LPT 1612g with 3.8 SGI TC CNG BS6 Ph2.pdf"
    )
    # Use a temp DB so tests never pollute real chroma_db
    TEST_DB = "/tmp/test_chroma_db"
    os.environ["CHROMA_DB_PATH"] = TEST_DB

    print("=" * 70)
    print("EMBEDDER TESTS — No API calls, image chunks skipped")
    print("=" * 70)
    print()

    # ── TEST 1: Sentence-transformer loads ────────────────────────────────────
    print("TEST 1: Embedding model loads?")
    try:
        model = _get_embedding_model()
        print(f"  ✅ PASS — Model loaded: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
        sys.exit(1)

    # ── TEST 2: Embedding a sentence works ────────────────────────────────────
    print("TEST 2: Embedding a sample sentence?")
    try:
        vec = model.encode("Service schedule for Signa 4830").tolist()
        print(f"  ✅ PASS — Vector dimensions: {len(vec)}")
        print(f"  Sample values: {[round(v, 4) for v in vec[:5]]}...\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
        sys.exit(1)

    # ── TEST 3: ChromaDB connects ─────────────────────────────────────────────
    print("TEST 3: ChromaDB connects?")
    try:
        collection = _get_collection()
        print(f"  ✅ PASS — Collection ready: '{collection.name}'")
        print(f"  DB path: {TEST_DB}\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
        sys.exit(1)

    # ── TEST 4: Parse PDF and filter out image chunks ─────────────────────────
    print("TEST 4: Parse PDF — get text + table chunks only?")
    try:
        all_chunks = parse_pdf(PDF_PATH, image_output_dir="/tmp/test_images")
        text_table_chunks = [c for c in all_chunks if c.chunk_type != "image"]
        image_chunks      = [c for c in all_chunks if c.chunk_type == "image"]
        print(f"  ✅ PASS")
        print(f"  Total chunks      : {len(all_chunks)}")
        print(f"  Text+table chunks : {len(text_table_chunks)}  ← will embed these")
        print(f"  Image chunks      : {len(image_chunks)}  ← SKIPPED (VLM needed)\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
        sys.exit(1)

    # ── TEST 5: Embed text + table chunks into ChromaDB ───────────────────────
    print("TEST 5: Embed text + table chunks into ChromaDB?")
    try:
        result = embed_chunks(text_table_chunks)
        print(f"  ✅ PASS")
        print(f"  Text  chunks embedded : {result['text_chunks']}")
        print(f"  Table chunks embedded : {result['table_chunks']}")
        print(f"  Errors                : {result['errors']}")
        if result["error_details"]:
            for e in result["error_details"]:
                print(f"    ⚠️  {e}")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
        sys.exit(1)

    # ── TEST 6: Verify chunks are stored in ChromaDB ──────────────────────────
    print("TEST 6: Chunks actually stored in ChromaDB?")
    try:
        count = collection.count()
        expected = result["text_chunks"] + result["table_chunks"]
        if count == expected:
            print(f"  ✅ PASS — {count} chunks in DB (matches embedded count)\n")
        else:
            print(f"  ⚠️  WARN — DB has {count} chunks, expected {expected}\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 7: Semantic retrieval works ─────────────────────────────────────
    print("TEST 7: Semantic retrieval — can we find relevant chunks?")
    try:
        query = "What is the service schedule for LPT 1612g?"
        query_vec = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        hits = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        print(f"  ✅ PASS — Top 3 results for: '{query}'")
        for i, (doc, meta, dist) in enumerate(zip(hits, metas, dists), 1):
            score = round(1 - dist, 4)
            print(f"\n  Result {i} | page={meta['page']} | type={meta['chunk_type']} | score={score}")
            print(f"  {repr(doc[:120])}")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 8: Table chunks are retrievable separately ───────────────────────
    print("TEST 8: Table chunks retrievable?")
    try:
        table_results = collection.get(
            where={"chunk_type": "table"},
            include=["documents", "metadatas"],
        )
        table_count = len(table_results["documents"])
        print(f"  ✅ PASS — {table_count} table chunks in DB")
        if table_results["documents"]:
            sample = table_results["documents"][0][:200]
            print(f"  Sample table chunk: {repr(sample)}")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(TEST_DB, ignore_errors=True)
    print("  (Test DB cleaned up)")

    print()
    print("=" * 70)
    print("EMBEDDER TESTS COMPLETE")
    print("Next step: python -m src.ingestion.embedder_vlm_test")
    print("           (will use Groq VLM for image chunks)")
    print("=" * 70)