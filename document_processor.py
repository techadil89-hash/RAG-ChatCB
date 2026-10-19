import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from llama_index.core.schema import Document

import config

DOCUMENT_METADATA_MAP: Dict[str, Dict[str, str]] = {
    "GROUP EMPLOYEE OPD BENEFITS PLAN BOOKLET 2024-25. (2).pdf": {
        "doc_title": "Group Employee OPD Benefits Plan Booklet (2024-25)",
        "category": "Employee Benefits & Health Insurance",
        "description": "Outpatient department (OPD) medical benefits guidelines, limits, claim submission, eligible dependents, and exclusions."
    },
    "Information Asset Protection Policy.pdf": {
        "doc_title": "EMEA Information Asset Protection Policy",
        "category": "Information Security & Asset Protection",
        "description": "Rules and classifications for safeguarding electronic networks, sensitive information assets, data classification, and authorized handling."
    },
    "Monitoring and Logging Policy.pdf": {
        "doc_title": "Monitoring and Logging Policy",
        "category": "IT Security & System Logging",
        "description": "Policy covering automated logging, monitoring of computer and network usage, log retention, reviews, and compliance requirements."
    }
}


def clean_text(text: str, file_name: str = "") -> str:
    """
    Cleans, normalizes, and removes OCR/PDF artifacts and repeated header boxes.
    """
    if not text:
        return ""

    # Normalize unicode representations
    text = unicodedata.normalize("NFKD", text)

    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")

    # Normalize Wingdings / PUA bullet symbols to standard bullets
    text = text.replace("\uf0b7", "• ").replace("\u2022", "• ")

    # Standardize typographical quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")

    # Document-specific repeated header / control box removal
    if "Monitoring and Logging Policy" in file_name:
        header_pattern = r"(?i)Docume\s*nt\s+Co\s*ntrol\s+Internal\s+Monitoring\s+and\s+Logging\s+Policy\s+Version\s+Number:\s*3\.0\s+Controlled\s+Document\s+not\s+to\s+be\s+amended\s+without\s+authority\s+of\s+policy\s+owner\s+Issue\s+Date\s*14/06/2016\s+Internal\s+and\s+not\s+for\s+reproduction\s+without\s+prior\s+consent\s+Page\s*\|\s*\d+"
        text = re.sub(header_pattern, "", text)

    if "Information Asset Protection Policy" in file_name:
        header_pattern = r"(?i)BCER\s*\|\s*EMEA\s+Information\s+Technology\s+EMEA\s+INFORMATION\s+ASSET\s+PROTECTION\s+POLICY"
        text = re.sub(header_pattern, "", text)

    # Fix broken hyphenated words at line breaks (e.g., "infor-\nmation" -> "information")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Standardize space indentation and multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Standardize line endings and collapse excessive newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def load_pdf(file_path: Path) -> List[Document]:
    """
    Extracts pages from a single PDF and returns LlamaIndex Document objects.
    Merges very short cover/title-only pages with subsequent content pages
    to prevent empty/isolated chunks.
    """
    reader = PdfReader(str(file_path))
    file_name = file_path.name
    meta_info = DOCUMENT_METADATA_MAP.get(file_name, {
        "doc_title": file_name,
        "category": "Corporate Documentation",
        "description": "Company policy or documentation."
    })

    raw_pages: List[Dict[str, Any]] = []
    total_pages = len(reader.pages)

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text, file_name=file_name)
        raw_pages.append({
            "page_num": page_idx + 1,
            "text": cleaned
        })

    # Page consolidation: if a page is very short (e.g. title-only cover < 80 chars),
    # carry it over to prepend onto the next page as context.
    consolidated_pages: List[Dict[str, Any]] = []
    carried_header = ""
    carried_page = None

    for item in raw_pages:
        text = item["text"]
        p_num = item["page_num"]

        if not text:
            continue

        if len(text) < 80 and p_num < total_pages:
            # Short cover/title page - carry forward to next page
            carried_header = text
            carried_page = p_num
            continue

        if carried_header:
            full_text = f"{carried_header}\n\n{text}"
            page_label = f"{carried_page}-{p_num}" if carried_page else str(p_num)
            carried_header = ""
            carried_page = None
        else:
            full_text = text
            page_label = str(p_num)

        consolidated_pages.append({
            "page_label": page_label,
            "text": full_text
        })

    # If the last page was short and was carried forward
    if carried_header:
        consolidated_pages.append({
            "page_label": str(carried_page or total_pages),
            "text": carried_header
        })

    documents: List[Document] = []
    for item in consolidated_pages:
        doc = Document(
            text=item["text"],
            metadata={
                "file_name": file_name,
                "doc_title": meta_info["doc_title"],
                "category": meta_info["category"],
                "page_number": item["page_label"],
                "total_pages": total_pages,
                "source": f"{file_name}#page={item['page_label']}",
            },
            excluded_embed_metadata_keys=["total_pages"],
            excluded_llm_metadata_keys=["total_pages"],
        )
        documents.append(doc)

    return documents


def load_all_target_pdfs() -> List[Document]:
    """
    Loads all target PDF files configured in config.py.
    """
    all_documents: List[Document] = []
    for pdf_name in config.TARGET_PDFS:
        pdf_path = config.DOCS_DIR / pdf_name
        if not pdf_path.exists():
            print(f"Warning: PDF file not found at {pdf_path}")
            continue

        docs = load_pdf(pdf_path)
        all_documents.extend(docs)

    return all_documents


if __name__ == "__main__":
    docs = load_all_target_pdfs()
    print(f"Successfully loaded {len(docs)} consolidated page documents.")
    for d in docs:
        print(f" - [{d.metadata['doc_title']} - Page {d.metadata['page_number']}] ({len(d.text)} chars)")
