"""
System health checks — run at startup and exposed via /health endpoint.
Validates parser, vector store, and model connectivity without API calls.
"""

import os
import time
from pathlib import Path

import fitz  # type: ignore[import]
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_START_TIME = time.time()


def check_parser() -> dict:
    """Verify PyMuPDF is working."""
    try:
        # Create a tiny in-memory PDF to test fitz
        doc = fitz.open()  # type: ignore[attr-defined]
        doc.new_page()     # type: ignore
        doc.close()
        return {"status": "ok", "detail": "PyMuPDF operational"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def check_vector_store() -> dict:
    """Verify ChromaDB is accessible and return chunk counts."""
    try:
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection("tata_motors_docs")
        count = collection.count()

        # Breakdown by chunk type
        breakdown = {"text": 0, "table": 0, "image": 0}
        if count > 0:
            results = collection.get(include=["metadatas"]) # type: ignore
            for meta in results["metadatas"]:               # type: ignore
                ctype = meta.get("chunk_type", "text")
                if ctype in breakdown:
                    breakdown[ctype] += 1

        warning = None
        if count == 0:
            warning = "No documents indexed yet — POST a PDF to /ingest first"
        elif breakdown["table"] == 0:
            warning = "No table chunks found — tables may not be parsing correctly"
        elif breakdown["image"] == 0:
            warning = "No image chunks found — images may not be extracting correctly"

        return {
            "status": "ok",
            "total_chunks": count,
            "breakdown": breakdown,
            "warning": warning,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def check_embedding_model() -> dict:
    """Verify sentence-transformer loads and can embed."""
    try:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        test_vec = model.encode("test").tolist()
        return {
            "status": "ok",
            "model": model_name,
            "vector_dimensions": len(test_vec),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def check_llm_connectivity() -> dict:
    """Verify Groq API key is set (does NOT make an actual API call)."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return {
            "status": "error",
            "detail": "GROQ_API_KEY not set — add to .env or Codespace secrets",
        }
    # Just check key exists and looks valid (starts with 'gsk_')
    if not groq_key.startswith("gsk_"):
        return {
            "status": "warning",
            "detail": "GROQ_API_KEY set but format looks unusual",
        }
    return {
        "status": "ok",
        "llm_model": os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"),
        "vlm_model": os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"),
    }


def run_all_checks() -> dict:
    """Run all health checks and return consolidated report."""
    checks = {
        "parser":          check_parser(),
        "vector_store":    check_vector_store(),
        "embedding_model": check_embedding_model(),
        "llm":             check_llm_connectivity(),
    }

    # Overall status — error if any check failed
    statuses = [c["status"] for c in checks.values()]
    if "error" in statuses:
        overall = "degraded"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "ok"

    return {
        "status": overall,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "checks": checks,
    }
