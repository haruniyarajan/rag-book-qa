"""
pdf_processor.py
----------------
Reads a PDF and splits it into smaller overlapping chunks.

Why chunks?
  LLMs can't read 300 pages at once. Splitting into ~500-word pieces
  lets us find the EXACT section that answers a question.

Why overlap?
  So sentences near chunk boundaries don't lose context.
"""

import os
import re
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from every page of a PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    file_name = os.path.basename(pdf_path)

    print(f"📖 Opening: {file_name}")
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"   Pages found: {total}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text or len(text.strip()) < 50:
                continue
            pages.append({
                "page_number": i + 1,
                "text": re.sub(r'\s+', ' ', text).strip(),
                "source": file_name,
            })

    print(f"   Readable pages: {len(pages)}")
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split pages into overlapping word-based chunks."""
    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()

        if len(words) <= chunk_size:
            chunks.append({
                "chunk_id":    f"p{page['page_number']}_c0",
                "text":        page["text"],
                "page_number": page["page_number"],
                "source":      page["source"],
                "chunk_index": chunk_id,
            })
            chunk_id += 1
            continue

        start = 0
        local_idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "chunk_id":    f"p{page['page_number']}_c{local_idx}",
                "text":        " ".join(words[start:end]),
                "page_number": page["page_number"],
                "source":      page["source"],
                "chunk_index": chunk_id,
            })
            chunk_id += 1
            local_idx += 1
            start += chunk_size - overlap

    print(f"   ✂️  Chunks created: {len(chunks)}")
    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Main function: PDF → pages → chunks."""
    print("\n=== 📄 PDF Processing ===")
    pages  = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    print("✅ Done.\n")
    return chunks
