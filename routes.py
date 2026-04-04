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

router = APIRouter()

# Track server start time for uptime reporting
_START_TIME = time.time()


# ── Pydantic Models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_chunks_indexed: int
    gemini_model: str
    embedding_model: str


class IngestResponse(BaseModel):
    filename: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_chunks: int
    processing_time_seconds: float
    errors: int


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


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
def health_check():
    """
    Returns system status including:
    - Model readiness
    - Number of indexed chunks
    - Server uptime
    """
    stats = get_index_stats()
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        total_chunks_indexed=stats["total_chunks"],
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
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

    Extracts:
    - **Text chunks** — paragraphs and headings
    - **Table chunks** — structured tables as text
    - **Image chunks** — images summarized via Gemini Vision, then embedded

    Returns a summary of ingested chunk counts and processing time.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    start_time = time.time()

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Parse PDF into chunks
        chunks = parse_pdf(tmp_path, image_output_dir="extracted_images")

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No content could be extracted from this PDF. "
                       "Ensure the file contains text, tables, or images.",
            )

        # Override source name to use original filename
        for chunk in chunks:
            chunk.source = file.filename

        # Embed all chunks into ChromaDB
        result = embed_chunks(chunks)

        processing_time = round(time.time() - start_time, 2)
        total = result["text_chunks"] + result["table_chunks"] + result["image_chunks"]

        return IngestResponse(
            filename=file.filename,
            text_chunks=result["text_chunks"],
            table_chunks=result["table_chunks"],
            image_chunks=result["image_chunks"],
            total_chunks=total,
            processing_time_seconds=processing_time,
            errors=result["errors"],
        )

    finally:
        # Clean up temp file
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

    Retrieves the most relevant text, table, and image-summary chunks,
    then generates a grounded answer using Gemini.

    Returns the answer with source references (filename, page, chunk type).
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    # Retrieve relevant chunks
    chunks = retrieve(query=request.question, top_k=request.top_k)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents indexed yet. Please POST a PDF to /ingest first.",
        )

    # Generate grounded answer
    answer = generate_answer(question=request.question, context_chunks=chunks)

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
    )
