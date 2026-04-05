"""
FastAPI route definitions.
Endpoints: /health, /ingest, /query, /docs (auto-generated)
"""

import os
import time
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from src.ingestion.parser import parse_pdf
from src.ingestion.embedder import embed_chunks, get_index_stats
from src.retrieval.retriever import retrieve
from src.models.gemini import generate_answer
from src.api.health import run_all_checks

router = APIRouter()


# ── Pydantic Models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    checks: dict


class IngestResponse(BaseModel):
    filename: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_chunks: int
    processing_time_seconds: float
    errors: int
    warnings: list[str]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the service schedule for Signa 4830?",
                    "top_k": 5,
                }
            ]
        }
    }


class SourceReference(BaseModel):
    source: str
    page: int
    chunk_type: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    chunks_retrieved: int
    system_warnings: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
def health_check():
    """
    Full system health report including:
    - PDF parser status
    - Vector store status + chunk breakdown (text/table/image)
    - Embedding model status
    - LLM/VLM API key check
    - Warnings if any chunk type is missing
    """
    report = run_all_checks()
    return HealthResponse(
        status=report["status"],
        uptime_seconds=report["uptime_seconds"],
        checks=report["checks"],
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a PDF document",
    tags=["Ingestion"],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF file to parse, embed, and index.

    Extracts text, table, and image chunks.
    Returns chunk counts with warnings if any type is missing.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    start_time = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = parse_pdf(tmp_path, image_output_dir="extracted_images")

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No content could be extracted from this PDF.",
            )

        for chunk in chunks:
            chunk.source = file.filename

        result = embed_chunks(chunks)

        # Smart warnings for missing chunk types
        warnings: list[str] = []
        if result["text_chunks"] == 0:
            warnings.append("No text chunks extracted — PDF may be scanned or image-only")
        if result["table_chunks"] == 0:
            warnings.append("No table chunks found — PDF may not contain tables")
        if result["image_chunks"] == 0:
            warnings.append("No image chunks found — PDF may not contain images")
        if result["errors"] > 0:
            warnings.append(
                f"{result['errors']} image(s) failed VLM summarization — "
                "stored as placeholders, may affect image query accuracy"
            )

        total = result["text_chunks"] + result["table_chunks"] + result["image_chunks"]

        return IngestResponse(
            filename=file.filename,
            text_chunks=result["text_chunks"],
            table_chunks=result["table_chunks"],
            image_chunks=result["image_chunks"],
            total_chunks=total,
            processing_time_seconds=round(time.time() - start_time, 2),
            errors=result["errors"],
            warnings=warnings,
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    tags=["Query"],
)
def query_documents(request: QueryRequest):
    """
    Ask a natural language question against all indexed documents.
    Returns grounded answer with source references and system warnings.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    chunks = retrieve(query=request.question, top_k=request.top_k)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents indexed yet. Please POST a PDF to /ingest first.",
        )

    answer = generate_answer(question=request.question, context_chunks=chunks)

    # Warn if retrieved chunks are all one type (low diversity = lower accuracy)
    chunk_types = set(c["chunk_type"] for c in chunks)
    warnings: list[str] = []
    if len(chunk_types) == 1:
        warnings.append(
            f"All retrieved chunks are of type '{list(chunk_types)[0]}' — "
            "answer may miss information from other modalities"
        )

    sources = [
        SourceReference(
            source=c["source"],
            page=c["page"],
            chunk_type=c["chunk_type"],
            relevance_score=c["score"],
        )
        for c in chunks
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        chunks_retrieved=len(chunks),
        system_warnings=warnings,
    )