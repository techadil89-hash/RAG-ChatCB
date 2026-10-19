import sys
import argparse

# Force UTF-8 stdout encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')

import config
from preview_chunks import preview_all_chunks
from indexer import build_or_update_index, query_index_test


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: PDF Document Ingestion, Chunking & Indexing"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview generated chunks and statistics without calling any API"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing (clears existing ChromaDB collection)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a test semantic retrieval query against the ChromaDB index"
    )

    args = parser.parse_args()

    # Step 1: Just preview
    if args.preview:
        preview_all_chunks(save_json=True)
        return

    # Step 2: Query existing index
    if args.query:
        query_index_test(args.query)
        return

    # Step 3: Run Full Indexing
    print("\n=======================================================")
    print("      STARTING RAG CHUNKING & INDEXING PIPELINE")
    print("=======================================================")
    print(f"Target Documents ({len(config.TARGET_PDFS)} files):")
    for f in config.TARGET_PDFS:
        print(f"  • {f}")

    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY.startswith("your_"):
        print("\n[NOTE] No GEMINI_API_KEY found in .env file.")
        print("Running chunking and inspection preview first...\n")
        preview_all_chunks(save_json=True)
        print("\n" + "=" * 65)
        print("NEXT STEP TO COMPLETE LIVE VECTOR INDEXING:")
        print("1. Open the file '.env' in this directory.")
        print("2. Enter your Google Gemini API key: GEMINI_API_KEY=AIzaSy...")
        print("3. Re-run: python run_indexing.py")
        print("=" * 65 + "\n")
        return

    # If API key is present, build the full vector index
    try:
        build_or_update_index(force_reindex=args.force)

        # Run a sample test query
        sample_query = "What medical outpatient expenses are covered under the OPD plan?"
        print("\nRunning automatic retrieval verification on indexed data:")
        query_index_test(sample_query, top_k=2)

    except Exception as e:
        print(f"\n[ERROR] An error occurred during indexing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
