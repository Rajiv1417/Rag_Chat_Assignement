"""
Multimodal RAG System — FastAPI Entry Point
Tata Motors Commercial Vehicle Aftersales Domain

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API docs available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Tata Motors Multimodal RAG API",
    description=(
        "A Retrieval-Augmented Generation system for querying Tata Motors "
        "commercial vehicle technical documents — service circulars, workshop "
        "manuals, ICGs, and ICMs — using natural language. "
        "Supports text, table, and image-based retrieval."
    ),
    version="1.0.0",
    contact={
        "name": "BITS WILP — Multimodal RAG Bootcamp Assignment",
    },
)

# Allow local frontend / Swagger UI calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Tata Motors Multimodal RAG API is running.",
        "docs": "/docs",
        "health": "/health",
    }
