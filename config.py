import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Google Gemini API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Fallback to Streamlit secrets if running inside Streamlit Cloud
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")

# Vector Database (ChromaDB) Settings
CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# Retrieval & Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Documents Directory
DOCS_DIR = BASE_DIR

# The 3 Target PDF Documents
TARGET_PDFS = [
    "GROUP EMPLOYEE OPD BENEFITS PLAN BOOKLET 2024-25. (2).pdf",
    "Information Asset Protection Policy.pdf",
    "Monitoring and Logging Policy.pdf",
]
