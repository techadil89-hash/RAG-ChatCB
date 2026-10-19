import os
import sys
import time
from typing import List, Optional
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

import config
from document_processor import load_all_target_pdfs
from chunker import create_chunks_from_documents, print_chunk_statistics


def get_chroma_client() -> chromadb.PersistentClient:
    """Initializes or connects to persistent ChromaDB storage."""
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_embedding_model() -> GoogleGenAIEmbedding:
    """
    Constructs GoogleGenAIEmbedding instance using the configured Gemini model
    and API key.
    """
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY.startswith("your_"):
        raise ValueError(
            "GEMINI_API_KEY is not configured or is using placeholder text.\n"
            "Please open .env file and set: GEMINI_API_KEY=your_actual_key\n"
            "Get your key from: https://aistudio.google.com/app/apikey"
        )

    return GoogleGenAIEmbedding(
        model_name=config.EMBEDDING_MODEL,
        api_key=config.GEMINI_API_KEY
    )


def build_or_update_index(
    nodes: Optional[List[BaseNode]] = None,
    force_reindex: bool = False
) -> VectorStoreIndex:
    """
    Indexes document chunks into ChromaDB with Gemini embeddings.
    """
    print("\n" + "=" * 65)
    print("                RAG INDEXING PIPELINE")
    print("=" * 65)

    chroma_client = get_chroma_client()

    # If force reindexing, delete existing collection first
    if force_reindex:
        try:
            chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
            print(f"[REINDEX] Cleared existing collection '{config.CHROMA_COLLECTION_NAME}'.")
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
    current_count = chroma_collection.count()
    print(f"ChromaDB Collection: '{config.CHROMA_COLLECTION_NAME}' (Current entries: {current_count})")

    if current_count > 0 and not force_reindex and nodes is None:
        print("[INFO] Index already exists. Connecting to existing vector store...")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embed_model = get_embedding_model()
        Settings.embed_model = embed_model
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

    # If nodes not supplied, load and chunk from target PDFs
    if nodes is None:
        print("\nStep 1: Loading PDF documents...")
        docs = load_all_target_pdfs()
        print(f"Loaded {len(docs)} document pages.")

        print("\nStep 2: Splitting into chunks...")
        nodes = create_chunks_from_documents(docs)
        print_chunk_statistics(nodes)

    # Initialize Embedding Model
    print("Step 3: Initializing Google Gemini Embedding Model...")
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    print(f"Using Embedding Model: {config.EMBEDDING_MODEL}")

    # Set up Chroma Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Generate Embeddings and Index
    print(f"\nStep 4: Embedding and indexing {len(nodes)} chunks into ChromaDB...")
    start_time = time.time()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    elapsed = round(time.time() - start_time, 2)

    print(f"\n[SUCCESS] Successfully indexed {len(nodes)} chunks in {elapsed}s!")
    print(f"ChromaDB collection count is now: {chroma_collection.count()}")
    print("=" * 65 + "\n")

    return index


def query_index_test(query_text: str, top_k: int = 3):
    """
    Executes a test retrieval query against the indexed ChromaDB collection.
    """
    print(f"\n>>> Running Test Semantic Search Query:")
    print(f"    Query: \"{query_text}\" (Top-K: {top_k})")
    
    chroma_client = get_chroma_client()
    chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query_text)

    print(f"\nFound {len(results)} relevant chunks:")
    for rank, node_with_score in enumerate(results, 1):
        score = getattr(node_with_score, "score", None)
        score_str = f"Score: {score:.4f}" if score is not None else ""
        node = node_with_score.node
        print(f"\n  [{rank}] {score_str} | Doc: {node.metadata.get('doc_title')} (Page {node.metadata.get('page_number')})")
        print(f"      Source: {node.metadata.get('file_name')}")
        snippet = node.text[:200].replace("\n", " ").strip()
        print(f"      Text: \"{snippet}...\"")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index PDF documents into ChromaDB with Gemini embeddings.")
    parser.add_argument("--force", action="store_true", help="Force reindexing of all documents")
    parser.add_argument("--test-query", type=str, default=None, help="Run a verification search query")
    args = parser.parse_args()

    try:
        if args.test_query:
            query_index_test(args.test_query)
        else:
            build_or_update_index(force_reindex=args.force)
    except ValueError as e:
        print(f"\n[CONFIGURATION NOTICE]\n{e}\n")
        print("To preview the chunks without an API key, run:")
        print("  python preview_chunks.py\n")
