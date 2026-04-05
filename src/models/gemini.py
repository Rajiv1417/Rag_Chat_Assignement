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
 
    Gemma 4 pattern (observed):
      *   Constraint 1: ...              ← bullet reasoning (REMOVE)
      *   Start: "Based on..."           ← bullet (REMOVE)
      Based on the provided document,... ← ACTUAL ANSWER (KEEP from here)
      Sources: [Source N]...             ← citation (KEEP)
 
    Priority:
      1. Find LAST "Based on the provided document" → take from there
      2. Find last non-bullet prose block (fallback)
      3. Return original (last resort)
    """
     # Step 1: Strip Gemma thinking tags
    text = re.sub(
        r"<\|channel\>thought.*?<channel\|>", "", text, flags=re.DOTALL
    ).strip()
 
    # Step 2: Find LAST occurrence of anchor phrase — take everything from there
    anchor = "Based on the provided document"
    last_idx = text.rfind(anchor)
    if last_idx != -1:
        return text[last_idx:].strip()

    # Step 3: Fallback — walk from end, find last non-bullet prose block
    def is_bullet(line: str) -> bool:
        s = line.strip()
        return bool(s and re.match(r"^(\*+|-|\d+\.)\s+", s))
 
    lines = text.split("\n")
    clean_start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue
        if is_bullet(lines[i]):
            break
        clean_start = i
 
    clean_block = "\n".join(lines[clean_start:]).strip()
    if clean_block and len(clean_block) > 20:
        return clean_block
 
    # Step 4: Last resort
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

    prompt = f"""You are an expert Tata Motors service assistant.
Answer the question using ONLY the provided context.
Do not show your thinking, reasoning, or bullet-point analysis.
Write your answer as clean prose paragraphs only.
Do not make up information not present in the context.
If the context does not contain enough information, say so clearly.
start  with: : "Based on the provided document, <answer>".
End with: "Sources: [Source N] filename, page X"
 
CONTEXT:
{context_text}
 
QUESTION: {question}
 
ANSWER:"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.1),
    )

    return _strip_thinking(response.text.strip())

