# Tata Motors Multimodal RAG System

> **BITS WILP — Multimodal RAG Bootcamp | Individual Assignment**

---

## Problem Statement

I work in the Commercial Vehicle Business Unit (CVBU) at Tata Motors, supporting aftersales operations across workshops spread across India. In this domain, a service advisor's ability to quickly retrieve accurate technical information directly determines the quality of customer service — and ultimately, vehicle uptime for fleet operators who depend on commercial vehicles for their livelihoods.

The challenge we face daily is not a lack of information. It is an overwhelming *abundance* of it, fragmented across thousands of documents in formats that were never designed to be queried together.

A Tata Motors service advisor today has access to over **2,600 service circulars** (PDFs), **1,500 ICG files** (Improvement Campaign Goodwill — free retrofit instructions in Word format), **1,500 ICM files** (Improvement Campaign Mandatory — design change records in Excel), and **180 workshop manuals** totalling upwards of 18,000 pages. Each of these document types is structurally different. Service circulars are dense PDFs with both narrative text and specification tables — for example, a service schedule for the LPO 1618 bus will contain maintenance intervals as structured tables alongside narrative paragraphs explaining conditions and exceptions. Workshop manuals are richly multimodal: they contain torque specification tables, step-by-step textual procedures, and critical engineering diagrams showing component assemblies, connector locations, and fluid circuit schematics. ICG files describe retrofit campaigns with eligibility defined by chassis number ranges, model families, and validity dates — information that combines tabular data with written scope descriptions.

A simple query like *"Is the EGR valve on chassis MAT828008L3C05118 covered under warranty?"* requires a service advisor to identify the vehicle model from the chassis number, look up its date of sale from the CRM, find the applicable warranty circular for emission components on BSVI vehicles, cross-reference the coverage table with the vehicle's age and kilometers, and then check whether any active ICG campaigns apply to that chassis range. Manually, this takes **20 to 30 minutes**. Across a busy day, advisors spend **2 to 3 hours** — nearly 25–30% of productive time — doing information retrieval instead of serving customers.

### Why This Is Not a Generic Document Q&A Problem

Most document Q&A systems are designed around text. Our domain breaks that assumption in three important ways.

First, **the most critical information lives in tables**, not paragraphs. Service schedules, torque specifications, warranty duration matrices, and ICG chassis eligibility ranges are all tabular. A system that extracts only plain text from these PDFs will systematically miss or corrupt the very data a service advisor needs most.

Second, **engineering diagrams are not decorative**. Workshop manuals include component location diagrams, wiring schematics, and assembly illustrations that are essential for a technician attempting a repair procedure. A text-only retrieval system cannot answer "where exactly is the DPF pressure sensor located on a Signa 4830 BSVI chassis?" — that answer exists only in an annotated diagram.

Third, **the terminology is deeply domain-specific and abbreviated**. Queries like "retro applicability for ICG_JSR_2026_01" or "brake bleeding procedure for Signa 4830" require understanding of Tata Motors-specific nomenclature — model families, emission standard suffixes (BSIV, BSVI), plant codes (JSR = Jamshedpur), and document ID conventions — that no general-purpose keyword search handles reliably.

Traditional keyword search fails because advisors do not know which circular to search. Fine-tuning a language model is impractical given the volume and continuous updates to the document corpus (new circulars are issued regularly). What is needed is a system that can retrieve the right multimodal chunks — text, tables, and image summaries — and synthesize a grounded, accurate answer in natural language.

### Expected Outcomes

A service advisor should be able to type *"What maintenance is overdue and which retros apply to chassis MAT828008L3C05118?"* and receive — within seconds — a synthesized answer drawing from a service circular (text and table), an ICG document (structured data), and a workshop manual diagram summary. This would reduce information lookup time from 15–30 minutes to under 2 minutes, improve ICG compliance rates, and allow new advisors to match the diagnostic quality of veterans — regardless of how long they have been in the role.

---

## Architecture Overview

```mermaid
graph TD
    A[PDF Upload via POST /ingest] --> B[PyMuPDF Parser]
    B --> C[Text Chunks]
    B --> D[Table Chunks]
    B --> E[Raw Images]
    E --> F[Gemma 4 Vision\nImage → Text Summary]
    F --> G[Image Summary Chunks]
    C --> H[sentence-transformers\nall-MiniLM-L6-v2]
    D --> H
    G --> H
    H --> I[(ChromaDB\nVector Store)]

    J[Natural Language Query\nPOST /query] --> K[Embed Query\nsentence-transformers]
    K --> L[Semantic Search\nChromaDB]
    I --> L
    L --> M[Top-K Chunks\ntext + table + image]
    M --> N[Gemma 4 LLM\nGrounded Answer Generation]
    N --> O[Answer + Source References]
```

---

## Technology Choices

| Component | Choice | Justification |
|---|---|---|
| **PDF Parser** | PyMuPDF (fitz) | Fastest PDF library in Python; handles text, tables, and image extraction reliably without a Java runtime (unlike Apache PDFBox). Docling was considered but adds heavy dependencies. |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Runs fully locally — no API cost, no rate limits. At 90MB, it fits comfortably in GitHub Codespaces (4GB RAM). Produces strong semantic embeddings for technical English. |
| **Vector Store** | ChromaDB (local persistent) | Zero-config local setup with a persistent on-disk store. No server process needed. Supports cosine similarity natively. Pinecone was considered but requires a paid plan for persistence. |
| **LLM** | Google Gemma 4| Free tier provides 1M tokens/day and 15 RPM — sufficient for development and evaluation. One API key covers both LLM and Vision tasks. |
| **VLM (Vision)** | Gemma 4 (same model) | Gemini's multimodal capability handles image description natively. Using the same model eliminates a second API key and simplifies configuration vs. a separate LLaVA or GPT-4o Vision setup. |
| **Framework** | FastAPI (direct, no LangChain) | LangChain adds abstraction overhead. Direct implementation with FastAPI + ChromaDB + Gemini gives cleaner, more debuggable code and demonstrates deeper understanding of the RAG pipeline. |
![Swagger UI Docs](Screenshots/01_swagger_ui.png)
![Health Check response](Screenshots/02_health_endpoint.png)
![Successful Ingestion Response](screenshots/03_ingest_response.png)
![Text Query Response](screenshots/04_text_query.png)
![Table Query Response](screenshots/05_table_query.png)
![Image Query Response](screenshots/06_image_query.png)