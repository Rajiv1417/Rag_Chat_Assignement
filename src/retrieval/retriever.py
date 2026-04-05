"""
Retriever — semantic search over the ChromaDB vector store.
Takes a natural language query, embeds it, and returns top-k relevant chunks.
"""

import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import chromadb

load_dotenv()

_embedding_model = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        client = chromadb.PersistentClient(path=db_path)
        _collection = client.get_or_create_collection(
            name="tata_motors_docs",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k semantically similar chunks for a query.

    Args:
        query: Natural language question from the user.
        top_k: Number of chunks to retrieve.

    Returns:
        List of dicts with keys: text, source, page, chunk_type, score.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    model = _get_embedding_model()
    query_embedding = model.encode(query).tolist()
    print("TEST: Semantic retrieval — can we find relevant chunks?")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"], # type: ignore
        )
        hits = results["documents"][0]      # type: ignore
        metas = results["metadatas"][0]     # type: ignore
        dists = results["distances"][0]     # type: ignore

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
            include=["documents", "metadatas"], # type: ignore
        )
        table_count = len(table_results["documents"])   # type: ignore
        print(f"  ✅ PASS — {table_count} table chunks in DB")
        if table_results["documents"]:
            sample = table_results["documents"][0][:200]
            print(f"  Sample table chunk: {repr(sample)}")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], # type: ignore
        results["metadatas"][0], # type: ignore
        results["distances"][0], # type: ignore
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "chunk_type": meta.get("chunk_type", "text"),
            "score": round(1 - dist, 4),  # cosine similarity (higher = better)
        })

    return chunks

   
