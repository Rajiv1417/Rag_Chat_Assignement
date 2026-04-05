import os
from dotenv import load_dotenv

from src.ingestion.parser import parse_pdf
from src.ingestion.embedder import embed_chunks, get_index_stats
from src.retrieval.retriever import retrieve
from src.models.gemini import generate_answer

load_dotenv()


def main():
    print("\n" + "=" * 70)
    print("🚀 FULL RAG PIPELINE TEST (WITH RETRIEVER)")
    print("=" * 70)

    pdf_path = "Documens/pdfs/SC_2025_36 Introduction of LPT 1612g with 3.8 SGI TC CNG BS6 Ph2.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found: {pdf_path}")
        return

    # ── STEP 1: PARSE PDF ─────────────────────────────
    print("\n📄 STEP 1: Parsing PDF...")
    chunks = parse_pdf(pdf_path)
    print(f"✅ Extracted {len(chunks)} chunks")

    # ── STEP 2: EMBEDDING ─────────────────────────────
    print("\n🧠 STEP 2: Embedding & storing...")
    result = embed_chunks(chunks)
    print("✅ Embedding result:", result)

    # ── STEP 3: DB STATS ─────────────────────────────
    stats = get_index_stats()
    print("\n📊 DB Stats:", stats)

    # ── STEP 4: QUERY TEST ────────────────────────────
    print("\n🔍 STEP 4: Retrieval + QA")

    test_questions = [
        "What engine is used in LPT 1612g?",
        "What is the service interval?",
        "Explain the diagram in the document",
    ]

    for q in test_questions:
        print("\n" + "-" * 60)
        print(f"❓ Question: {q}")

        retrieved_chunks = retrieve(q, top_k=3)

        if not retrieved_chunks:
            print("⚠️ No chunks retrieved")
            continue

        print(f"\n📦 Retrieved {len(retrieved_chunks)} chunks")

        answer = generate_answer(q, retrieved_chunks)

        print("\n💡 ANSWER:")
        print(answer)

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()