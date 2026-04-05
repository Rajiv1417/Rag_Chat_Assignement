"""
VLM Test — tests Gemma 4 31B Vision on 2 images only.
Run after embedder tests pass.

Usage:
    PYTHONPATH=/workspaces/Rag_Chat_Assignement \
    ORT_LOGGING_LEVEL=3 \
    /workspaces/Rag_Chat_Assignement/bootcamp/bin/python \
    -m src.ingestion.vlm_test
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PDF_PATH = os.getenv(
    "TEST_PDF",
    "/workspaces/Rag_Chat_Assignement/Documens/pdfs/"
    "SC_2025_36 Introduction of LPT 1612g with 3.8 SGI TC CNG BS6 Ph2.pdf"
)
IMAGE_DIR = "/tmp/vlm_test_images"
MAX_IMAGES_TO_TEST = 2  # only 2 API calls — saves rate limit

print("=" * 70)
print("VLM TESTS — Gemma 4 31B Vision via Google AI Studio (2 images only)")
print("=" * 70)
print()

# ── TEST 1: GEMINI_API_KEY is set ─────────────────────────────────────────────
print("TEST 1: GEMINI_API_KEY is set?")
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("  ❌ FAIL — GEMINI_API_KEY not set")
    print("  Add to .env or Codespace Secrets")
    sys.exit(1)
masked = api_key[:6] + "..." + api_key[-4:]
print(f"  ✅ PASS — Key found: {masked}\n")

# ── TEST 2: Google Generative AI client initializes ───────────────────────────
print("TEST 2: Google AI client initializes?")
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
    model = genai.GenerativeModel(model_name)
    print(f"  ✅ PASS — Model ready: {model_name}\n")
except Exception as e:
    print(f"  ❌ FAIL — {e}")
    print("  Run: pip install google-generativeai")
    sys.exit(1)

# ── TEST 3: Quick text call (no image, no rate limit risk) ────────────────────
print("TEST 3: Basic text generation works?")
try:
    response = model.generate_content("Reply with exactly one word: Ready")
    print(f"  ✅ PASS — Response: {repr(response.text.strip())}\n")
except Exception as e:
    err = str(e)
    if "429" in err:
        print(f"  ⚠️  RATE LIMIT — {err[:120]}")
        print("  Wait a few minutes and retry")
        sys.exit(1)
    else:
        print(f"  ❌ FAIL — {err[:200]}\n")
        sys.exit(1)

# ── TEST 4: Extract images from PDF ───────────────────────────────────────────
print("TEST 4: Extract images from PDF?")
try:
    from src.ingestion.parser import parse_pdf
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    all_chunks = parse_pdf(PDF_PATH, image_output_dir=IMAGE_DIR)
    image_chunks = [c for c in all_chunks if c.chunk_type == "image"]
    print(f"  ✅ PASS — {len(image_chunks)} image chunks found")
    print(f"  Testing only first {MAX_IMAGES_TO_TEST} to save rate limit\n")
except Exception as e:
    print(f"  ❌ FAIL — {e}")
    sys.exit(1)

# ── TEST 5: Gemma Vision summarizes images ────────────────────────────────────
print("TEST 5: Gemma 4 31B Vision summarizes images?")

from src.models.gemini import summarize_image

test_chunks = image_chunks[:MAX_IMAGES_TO_TEST]
success = 0
last_summary = ""

for i, chunk in enumerate(test_chunks, 1):
    print(f"\n  Image {i}/{MAX_IMAGES_TO_TEST} — page {chunk.page}")
    print(f"  File: {chunk.image_path}")
    try:
        summary = summarize_image(chunk.image_path) # type: ignore
        last_summary = summary
        print(f"  ✅ PASS — {len(summary)} chars returned")
        print(f"  Preview: {repr(summary[:250])}")
        success += 1
        if i < MAX_IMAGES_TO_TEST:
            print(f"\n  Waiting 3s before next call...")
            time.sleep(3)
    except Exception as e:
        err = str(e)
        if "429" in err:
            print(f"  ⚠️  RATE LIMIT — {err[:150]}")
            print(f"  Try again in a few minutes")
        elif "400" in err:
            print(f"  ❌ IMAGE FORMAT ERROR — {err[:150]}")
            print(f"  Model may not support this image type")
        else:
            print(f"  ❌ FAIL — {err[:200]}")

print()

# ── TEST 6: VLM summary is embeddable ────────────────────────────────────────
if last_summary:
    print("TEST 6: VLM summary can be embedded?")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        vec = embed_model.encode(f"[IMAGE DESCRIPTION] {last_summary}").tolist()
        print(f"  ✅ PASS — Embedded into {len(vec)}-dim vector\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
else:
    print("TEST 6: SKIPPED — No successful VLM summaries\n")

# ── Cleanup ───────────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(IMAGE_DIR, ignore_errors=True)
print("  (Test images cleaned up)")

# ── Final verdict ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
if success == MAX_IMAGES_TO_TEST:
    print(f"✅ VLM TESTS COMPLETE — {success}/{MAX_IMAGES_TO_TEST} images passed")
    print("Next step: bash run.sh  — start the full FastAPI server!")
elif success > 0:
    print(f"⚠️  PARTIAL — {success}/{MAX_IMAGES_TO_TEST} images passed")
    print("Check errors above before running full pipeline")
else:
    print(f"❌ FAILED — 0/{MAX_IMAGES_TO_TEST} images passed")
    print("Fix errors above before proceeding")
print("=" * 70)