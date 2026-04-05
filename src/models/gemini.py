"""
Gemini model wrapper — handles both LLM (text) and VLM (image) tasks.
Single API key, single model, zero cost on free tier.
"""

import os
import base64
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def _get_client() -> genai.GenerativeModel:
    """Initialize and return Gemini model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env file")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)


def summarize_image(image_path: str) -> str:
    """
    Send an image to Gemini Vision and get a descriptive text summary.
    Used during ingestion to convert PDF images into searchable text chunks.

    Args:
        image_path: Path to the image file (PNG/JPEG).

    Returns:
        Text description of the image content.
    """
    model = _get_client()

    with open(image_path, "rb") as f:
        image_data = f.read()

    prompt = (
        "You are analyzing a page image from a Tata Motors commercial vehicle "
        "technical document. Describe everything you see in detail: "
        "any diagrams, charts, tables, component labels, measurements, "
        "part numbers, arrows, annotations, and figures. "
        "Be specific and technical — this description will be used to answer "
        "maintenance and repair queries from service advisors. "
        "If the image contains a table, reproduce its contents in text form. "
        "If it is an engineering diagram, describe each labeled component and its relationship."
    )

    response = model.generate_content([
        {"mime_type": "image/png", "data": image_data},
        prompt
    ])

    return response.text.strip()


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generate a grounded answer using retrieved context chunks.

    Args:
        question: User's natural language query.
        context_chunks: List of dicts with keys: text, source, page, chunk_type.

    Returns:
        Grounded answer string with source references.
    """
    model = _get_client()

    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += (
            f"\n[Source {i}] File: {chunk['source']} | "
            f"Page: {chunk['page']} | Type: {chunk['chunk_type']}\n"
            f"{chunk['text']}\n"
            f"{'—' * 60}\n"
        )

    prompt = f"""You are an expert Tata Motors service assistant. 
Answer the question using ONLY the provided context. 
Do not make up information not present in the context.
If the context does not contain enough information, say so clearly.

After your answer, list the sources you used as:
"Sources: [Source N] filename, page X"

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

    response = model.generate_content(prompt)
    return response.text.strip()
