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
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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


if __name__ == "__main__":
    print("Available Generative AI Models:")
    for m in genai.list_models():
        if "embedContent" in m.supported_generation_methods:
            print(f"  - {m.name} (Embeddings)")
        else:
            print(f"  - {m.name}")
    # -------- TEST 1: Image Summarization --------
    try:
        test_image_path = "/workspaces/Rag_Chat_Assignement/Documens/images/Service Circular LPT 1612g.png"  # put any test image here

        if os.path.exists(test_image_path):
            print("\n--- IMAGE SUMMARY TEST ---\n")
            summary = summarize_image(test_image_path)
            print(summary)
        else:
            print("\n[SKIP] sample.png not found for image test")

    except Exception as e:
        print(f"\n[ERROR - IMAGE TEST]: {e}")


    # -------- TEST 2: Q&A with Context --------
    try:
        print("\n--- QA TEST ---\n")

        sample_context = [
            {
                "text": "The LPT 1612g CNG BS6 Phase 2 truck uses a 3.8 SGI TC engine with improved fuel efficiency and reduced emissions.",
                "source": "LPT_1612g_doc.pdf",
                "page": 3,
                "chunk_type": "text"
            },
            {
                "text": "Maintenance interval for engine oil is 20,000 km under standard operating conditions.",
                "source": "LPT_1612g_doc.pdf",
                "page": 12,
                "chunk_type": "table"
            }
        ]

        question = "What engine is used in LPT 1612g and what is the service interval?"

        answer = generate_answer(question, sample_context)
        print(answer)

    except Exception as e:
        print(f"\n[ERROR - QA TEST]: {e}")