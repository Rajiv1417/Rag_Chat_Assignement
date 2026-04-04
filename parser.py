"""
PDF Parser — extracts three chunk types from any PDF:
  1. Text chunks   (paragraphs, headings)
  2. Table chunks  (structured table content as markdown-style text)
  3. Image chunks  (raw images saved to disk, later summarized by VLM)
"""

import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF


ChunkType = Literal["text", "table", "image"]


@dataclass
class ParsedChunk:
    """A single extracted chunk from a PDF page."""
    text: str
    chunk_type: ChunkType
    source: str          # original filename
    page: int
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_path: str | None = None   # only set for image chunks


def _is_table_block(block: dict) -> bool:
    """
    Heuristic: detect if a text block looks like a table.
    Checks for multiple tab-separated or pipe-separated columns,
    or lines with consistent numeric column patterns.
    """
    lines = block.get("lines", [])
    if len(lines) < 2:
        return False

    tab_lines = sum(1 for line in lines if "\t" in _line_text(line))
    multi_span = sum(1 for line in lines if len(line.get("spans", [])) >= 3)

    return tab_lines >= 2 or multi_span >= (len(lines) * 0.6)


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line.get("spans", []))


def _block_to_table_text(block: dict) -> str:
    """Convert a table-like block into a readable text representation."""
    rows = []
    for line in block.get("lines", []):
        row = " | ".join(span["text"].strip() for span in line.get("spans", []) if span["text"].strip())
        if row:
            rows.append(row)
    return "\n".join(rows)


def parse_pdf(pdf_path: str, image_output_dir: str = "extracted_images") -> list[ParsedChunk]:
    """
    Parse a PDF and return a list of ParsedChunk objects.

    Args:
        pdf_path:         Path to the PDF file.
        image_output_dir: Directory to save extracted images.

    Returns:
        List of ParsedChunk objects (text + table + image chunks).
    """
    pdf_path = Path(pdf_path)
    source_name = pdf_path.name
    image_dir = Path(image_output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[ParsedChunk] = []

    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc, start=1):
        # ── 1. Extract text and table blocks ──────────────────────────
        blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        text_buffer: list[str] = []

        for block in blocks:
            if block["type"] != 0:  # 0 = text block
                continue

            if _is_table_block(block):
                # Flush any accumulated text first
                if text_buffer:
                    combined = " ".join(text_buffer).strip()
                    if len(combined) > 30:
                        chunks.append(ParsedChunk(
                            text=combined,
                            chunk_type="text",
                            source=source_name,
                            page=page_num,
                        ))
                    text_buffer = []

                table_text = _block_to_table_text(block)
                if table_text.strip():
                    chunks.append(ParsedChunk(
                        text=f"[TABLE]\n{table_text}",
                        chunk_type="table",
                        source=source_name,
                        page=page_num,
                    ))
            else:
                # Accumulate plain text
                for line in block.get("lines", []):
                    line_text = _line_text(line).strip()
                    if line_text:
                        text_buffer.append(line_text)

                # Flush at paragraph boundaries (blank line between blocks)
                if text_buffer and len(" ".join(text_buffer)) > 200:
                    combined = " ".join(text_buffer).strip()
                    chunks.append(ParsedChunk(
                        text=combined,
                        chunk_type="text",
                        source=source_name,
                        page=page_num,
                    ))
                    text_buffer = []

        # Flush remaining text
        if text_buffer:
            combined = " ".join(text_buffer).strip()
            if len(combined) > 30:
                chunks.append(ParsedChunk(
                    text=combined,
                    chunk_type="text",
                    source=source_name,
                    page=page_num,
                ))

        # ── 2. Extract images ──────────────────────────────────────────
        image_list = page.get_images(full=True)
        for img_index, img_ref in enumerate(image_list):
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            # Skip tiny images (logos, borders) — less than 5KB
            if len(image_bytes) < 5120:
                continue

            img_filename = f"{source_name}_p{page_num}_img{img_index}.{ext}"
            img_path = image_dir / img_filename

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Text placeholder — will be replaced with VLM summary during embedding
            chunks.append(ParsedChunk(
                text="[IMAGE — awaiting VLM summary]",
                chunk_type="image",
                source=source_name,
                page=page_num,
                image_path=str(img_path),
            ))

    doc.close()
    return chunks
