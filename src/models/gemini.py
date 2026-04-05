"""
LLM + VLM wrapper using Google AI Studio API with Gemma 4 31B.

Functions:
  summarize_image()  — Gemma Vision: image → detailed text description
  generate_answer()  — Gemma LLM: retrieved chunks + question → grounded answer
"""

import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def _get_model() -> genai.GenerativeModel:
    """Initialize Gemma model via Google AI Studio."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env or Codespace Secrets")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
    return genai.GenerativeModel(model_name)


def _strip_thinking(text: str) -> str:
    """
    Extract actual answer from Gemma 4's response.

    Gemma 4 consistently wraps its answer in patterns like:
      - "**Analyze the image:** <actual content>"
      - "**Final Answer:** <actual content>"
      - "Final Answer: <actual content>"

    Priority order:
      1. Look for known Gemma 4 answer markers → take content after them
      2. Strip Gemma 4 thinking tags
      3. Fallback: return original text
    """
    # ── Priority 1: Gemma 4 answer markers ───────────────────────────────────
    answer_markers = [
        r"\*\*Analyze the image:\*\*",   # **Analyze the image:**
        r"\*\*Final Answer:\*\*",         # **Final Answer:**
        r"Final Answer:",                  # Final Answer:
        r"\*\*Answer:\*\*",               # **Answer:**
        r"\*\*Description:\*\*",          # **Description:**
        r"\*\*Response:\*\*",             # **Response:**
    ]

    for marker in answer_markers:
        match = re.search(marker, text)
        if match:
            after = text[match.end():].strip()
            if after:
                return after

    # ── Priority 2: Gemma 4 thinking tags ────────────────────────────────────
    cleaned = re.sub(
        r"<\|channel\>thought.*?<channel\|>", "", text, flags=re.DOTALL
    ).strip()
    if cleaned and cleaned != text:
        return cleaned

    # ── Priority 3: Fallback — return as-is ──────────────────────────────────
    return text


def summarize_image(image_path: str) -> str:
    """
    Send an image to Gemma 4 Vision and get a detailed text description.
    Used during ingestion to convert PDF images into searchable text chunks.

    Args:
        image_path: Path to the image file (PNG/JPEG).

    Returns:
        Detailed technical description of the image (thinking stripped).
    """
    model = _get_model()

    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = image_path.lower().split(".")[-1]
    mime_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    prompt = (
        "Directly describe this Tata Motors technical document image. "
        "Do not show your thinking or reasoning. Output only the description.\n\n"
        "Cover all of the following that are present:\n"
        "- Diagrams: name each labeled component and describe its function\n"
        "- Tables: reproduce every row and column as plain text\n"
        "- Visible text: part numbers, specifications, measurements, warnings\n"
        "- Charts or graphs: describe axes, values, and key trends\n"
        "- Vehicle components: use technical names\n\n"
        "Be specific and technical. This will be used by service advisors "
        "to answer maintenance and repair queries."
    )

    response = model.generate_content(
        [{"mime_type": mime_type, "data": image_data}, prompt],
        generation_config=genai.GenerationConfig(temperature=0.1),
    )

    return _strip_thinking(response.text.strip())


def generate_answer(question: str, context_chunks: list) -> str:
    """
    Generate a grounded answer using retrieved context chunks.

    Args:
        question: User's natural language query.
        context_chunks: List of dicts with keys: text, source, page, chunk_type.

    Returns:
        Grounded answer with source references (thinking stripped).
    """
    model = _get_model()

    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += (
            f"\n[Source {i}] File: {chunk['source']} | "
            f"Page: {chunk['page']} | Type: {chunk['chunk_type']}\n"
            f"{chunk['text']}\n"
            f"{'—' * 60}\n"
        )

    prompt = f"""
You are a Tata Motors service assistant.

Answer ONLY using the provided context.

Rules:
- Do NOT explain your reasoning
- Do NOT repeat instructions
- Do NOT mention context or sources in explanation
- If answer not found, say: "Information not available in provided documents."

At the end, add:
Sources: [Source N] filename, page X

CONTEXT:
{context_text}

QUESTION: {question}

FINAL ANSWER:
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.2),
    )

    return _strip_thinking(response.text.strip())

