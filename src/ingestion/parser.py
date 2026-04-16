"""
PDF Parser — extracts three chunk types from any PDF:
  1. Text chunks   (paragraphs, headings)
  2. Table chunks  (detected via PyMuPDF's native find_tables())
  3. Image chunks  (raw images saved to disk, later summarized by VLM)
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF 1.23+


ChunkType = Literal["text", "table", "image"]


@dataclass
class ParsedChunk:
    """A single extracted chunk from a PDF page."""
    text: str
    chunk_type: ChunkType
    source: str
    page: int
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_path: str | None = None

def _table_to_text(table_data: list) -> str:
    """
    Convert PyMuPDF table (list of rows) to pipe-separated text.
    e.g. [["Col1", "Col2"], ["Val1", "Val2"]] → "Col1 | Col2\nVal1 | Val2"
    """
    rows = []
    for row in table_data:
        cleaned = [str(cell).strip() if cell else "" for cell in row]
        if any(cleaned):  # skip completely empty rows
            rows.append(" | ".join(cleaned))
    return "\n".join(rows)


def _get_table_bboxes(page) -> list:  # type: ignore[type-arg]
    """Return bounding boxes of all detected tables on this page."""
    try:
        tabs = page.find_tables()  # type: ignore[attr-defined]
        return [t.bbox for t in tabs.tables]
    except Exception:
        return []


def _bbox_contains(table_bbox: tuple, block_bbox: tuple) -> bool:
    """Check if a block falls inside a table bounding box (with tolerance)."""
    tx0, ty0, tx1, ty1 = table_bbox
    bx0, by0, bx1, by1 = block_bbox
    tolerance = 5
    return (bx0 >= tx0 - tolerance and by0 >= ty0 - tolerance and
            bx1 <= tx1 + tolerance and by1 <= ty1 + tolerance)

def check_pdf_path(pdf_path: str):
    """Validate PDF path and raise exceptions if invalid."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not Path(pdf_path).is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    if Path(pdf_path).suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")


def parse_pdf(pdf_path: str, image_output_dir: str = "extracted_images") -> list[ParsedChunk]:
    """
    Parse a PDF and return a list of ParsedChunk objects.

    Args:
        pdf_path:         Path to the PDF file.
        image_output_dir: Directory to save extracted images.

    Returns:
        List of ParsedChunk objects (text + table + image chunks).
    """
    pdf = Path(pdf_path)
    source_name = pdf.name
    image_dir = Path(image_output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[ParsedChunk] = []
    doc = fitz.open(str(pdf))  # type: ignore[attr-defined]

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-indexed

        # ── 1. Detect tables using PyMuPDF native method ──────────────
        table_chunks_this_page: list[ParsedChunk] = []
        table_bboxes: list = []

        try:
            tabs = page.find_tables()  # type: ignore[attr-defined]
            for table in tabs.tables:
                table_text = table.to_markdown()
                if table_text.strip():
                    table_chunks_this_page.append(ParsedChunk(
                        text=f"[TABLE]\n{table_text}",
                        chunk_type="table",
                        source=source_name,
                        page=page_num,
                    ))
                    table_bboxes.append(table.bbox)
        except Exception:
            pass  # find_tables() not available or failed — skip table detection

        chunks.extend(table_chunks_this_page)

        # ── 2. Extract text blocks (skip blocks inside tables) ────────
        blocks = page.get_text("blocks")  # type: ignore[attr-defined]
        # blocks format: (x0, y0, x1, y1, text, block_no, block_type)

        text_buffer: list[str] = []

        for block in blocks:
            block_type = block[6]
            if block_type != 0:  # 0 = text, 1 = image
                continue

            block_bbox = (block[0], block[1], block[2], block[3])
            block_text = block[4].strip()

            if not block_text:
                continue

            # Skip text blocks that are inside a detected table
            if any(_bbox_contains(tb, block_bbox) for tb in table_bboxes):
                continue

            text_buffer.append(block_text)

            # Flush buffer when it's large enough to be a meaningful chunk
            if len(" ".join(text_buffer)) > 300:
                chunks.append(ParsedChunk(
                    text=" ".join(text_buffer).strip(),
                    chunk_type="text",
                    source=source_name,
                    page=page_num,
                ))
                text_buffer = []

        # Flush remaining text for this page
        if text_buffer:
            combined = " ".join(text_buffer).strip()
            if len(combined) > 20:
                chunks.append(ParsedChunk(
                    text=combined,
                    chunk_type="text",
                    source=source_name,
                    page=page_num,
                ))

        # ── 3. Extract images ──────────────────────────────────────────
        for img_idx, img_ref in enumerate(page.get_images(full=True)):  # type: ignore[attr-defined]
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            # Skip tiny images — logos, borders, decorations (<5KB)
            if len(image_bytes) < 5120:
                continue

            img_filename = f"{source_name}_p{page_num}_img{img_idx}.{ext}"
            img_path = image_dir / img_filename

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            chunks.append(ParsedChunk(
                text="[IMAGE — awaiting VLM summary]",
                chunk_type="image",
                source=source_name,
                page=page_num,
                image_path=str(img_path),
            ))

    doc.close()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# PARSER TESTS — run directly: python -m src.ingestion.parser
# No API calls, no embeddings — only PDF parsing logic tested here.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os

    # ── Config ────────────────────────────────────────────────────────────────
    PDF_PATH = os.getenv(
        "TEST_PDF",
        "/workspaces/Rag_Chat_Assignement/Documens/pdfs/"
        "SC_2025_36 Introduction of LPT 1612g with 3.8 SGI TC CNG BS6 Ph2.pdf"
    )

    print("=" * 70)
    print("PARSER TESTS — No API calls, pure PDF parsing only")
    print("=" * 70)
    print(f"PDF: {PDF_PATH}\n")

    # ── TEST 1: File exists ───────────────────────────────────────────────────
    print("TEST 1: File exists?")
    if not Path(PDF_PATH).exists():
        print(f"  ❌ FAIL — File not found: {PDF_PATH}")
        print(f"  Set TEST_PDF env var to correct path:")
        print(f"  TEST_PDF='your/path.pdf' python -m src.ingestion.parser")
        sys.exit(1)
    print(f"  ✅ PASS — File found ({Path(PDF_PATH).stat().st_size // 1024} KB)\n")

    # ── TEST 2: PDF opens correctly ───────────────────────────────────────────
    print("TEST 2: PDF opens correctly?")
    try:
        doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
        page_count = len(doc)
        doc.close()
        print(f"  ✅ PASS — {page_count} pages found\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")
        sys.exit(1)

    # ── TEST 3: Raw text extraction (page 1) ──────────────────────────────────
    print("TEST 3: Raw text extraction from page 1?")
    try:
        doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
        page = doc[0]
        raw_text = page.get_text().strip()  # type: ignore[attr-defined]
        doc.close()
        if raw_text:
            print(f"  ✅ PASS — Text found on page 1")
            print(f"  Sample (first 200 chars): {repr(raw_text[:200])}\n")
        else:
            print(f"  ⚠️  WARN — No text on page 1 (might be scanned image-only page)\n")
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 4: Block-level breakdown (page 1) ────────────────────────────────
    print("TEST 4: Block types on page 1?")
    try:
        doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
        page = doc[0]
        blocks = page.get_text("blocks")  # type: ignore[attr-defined]
        text_blocks  = [b for b in blocks if b[6] == 0]
        image_blocks = [b for b in blocks if b[6] == 1]
        doc.close()
        print(f"  ✅ PASS")
        print(f"  Total blocks : {len(blocks)}")
        print(f"  Text  blocks : {len(text_blocks)}")
        print(f"  Image blocks : {len(image_blocks)}")
        if text_blocks:
            sample = text_blocks[0][4].strip()[:150]
            print(f"  First text block sample: {repr(sample)}")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 5: Native table detection (page 1) ───────────────────────────────
    print("TEST 5: Native table detection on page 1?")
    try:
        doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
        page = doc[0]
        tabs = page.find_tables()  # type: ignore[attr-defined]
        table_count = len(tabs.tables)
        doc.close()
        if table_count > 0:
            print(f"  ✅ PASS — {table_count} table(s) found on page 1")
            doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
            page = doc[0]
            tabs = page.find_tables()  # type: ignore[attr-defined]
            sample_data = tabs.tables[0].extract()
            doc.close()
            print(f"  Table rows: {len(sample_data)}, cols: {len(sample_data[0]) if sample_data else 0}")
            print(f"  First row: {sample_data[0] if sample_data else 'empty'}")
        else:
            print(f"  ⚠️  WARN — No tables on page 1 (check other pages)")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 6: Table detection across ALL pages ───────────────────────────────
    print("TEST 6: Table detection across all pages?")
    try:
        doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
        total_tables = 0
        pages_with_tables = []
        for i in range(len(doc)):
            tabs = doc[i].find_tables()  # type: ignore[attr-defined]
            if tabs.tables:
                total_tables += len(tabs.tables)
                pages_with_tables.append(i + 1)
        doc.close()
        if total_tables > 0:
            print(f"  ✅ PASS — {total_tables} table(s) found across PDF")
            print(f"  Pages with tables: {pages_with_tables}")
        else:
            print(f"  ⚠️  WARN — No tables found in entire PDF")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 7: Image extraction (page 1) ────────────────────────────────────
    print("TEST 7: Image extraction from page 1?")
    try:
        doc = fitz.open(PDF_PATH)  # type: ignore[attr-defined]
        page = doc[0]
        images = page.get_images(full=True)  # type: ignore[attr-defined]
        large_images = []
        for img_ref in images:
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            size = len(base_image["image"])
            if size >= 5120:
                large_images.append((base_image["ext"], size // 1024))
        doc.close()
        print(f"  ✅ PASS — {len(images)} image(s) on page 1, "
              f"{len(large_images)} above 5KB threshold")
        for ext, kb in large_images:
            print(f"    → .{ext} | {kb} KB")
        print()
    except Exception as e:
        print(f"  ❌ FAIL — {e}\n")

    # ── TEST 8: Full parse_pdf() function ─────────────────────────────────────
    print("TEST 8: Full parse_pdf() — all pages, all chunk types?")
    try:
        chunks = parse_pdf(PDF_PATH, image_output_dir="/tmp/test_extracted_images")
        text_c  = [c for c in chunks if c.chunk_type == "text"]
        table_c = [c for c in chunks if c.chunk_type == "table"]
        image_c = [c for c in chunks if c.chunk_type == "image"]

        print(f"  ✅ PASS — Total chunks: {len(chunks)}")
        print(f"  Text  chunks : {len(text_c)}")
        print(f"  Table chunks : {len(table_c)}")
        print(f"  Image chunks : {len(image_c)}")

        if text_c:
            print(f"\n  Sample TEXT chunk (page {text_c[0].page}):")
            print(f"  {repr(text_c[0].text[:200])}")

        if table_c:
            print(f"\n  Sample TABLE chunk (page {table_c[0].page}):")
            print(f"  {table_c[0].text[:300]}")

        if image_c:
            print(f"\n  Sample IMAGE chunk (page {image_c[0].page}):")
            print(f"  saved to: {image_c[0].image_path}")

        # Final verdict
        print()
        issues = []
        if not text_c:
            issues.append("No text chunks — check if PDF is scanned")
        if not table_c:
            issues.append("No table chunks — PDF may not have tables")
        if not image_c:
            issues.append("No image chunks — PDF may not have images")

        if issues:
            print("  ⚠️  Warnings:")
            for w in issues:
                print(f"    - {w}")
        else:
            print("  🎉 All 3 chunk types found — parser is working correctly!")
            print("  ✅ Safe to proceed to embedding step.")

    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("PARSER TESTS COMPLETE")
    print("=" * 70)