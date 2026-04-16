"""
Multimodal RAG System — FastAPI Entry Point
Tata Motors Commercial Vehicle Aftersales Domain

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API docs available at:
    http://localhost:8000/docs
"""
import sys
import os
 
# Ensure project root is in Python path (required for Codespaces / non-package runs)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

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

 
# Serve frontend UI
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
 
    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {
            "message": "Tata Motors Multimodal RAG API is running.",
            "ui": "Add frontend/index.html to enable UI",
            "docs": "/docs",
        }
 