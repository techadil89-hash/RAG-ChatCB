import sys
import json
from pathlib import Path
from tabulate import tabulate

# Force UTF-8 encoding on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

import config
from document_processor import load_all_target_pdfs
from chunker import create_chunks_from_documents, get_chunking_statistics


def preview_all_chunks(save_json: bool = True):
    """
    Loads all target documents, processes them, generates chunks,
    and displays an interactive and detailed inspection report.
    """
    print("\n" + "=" * 70)
    print("           RAG CHATBOT: DOCUMENT CHUNKING & PREVIEW")
    print("=" * 70)
    print(f"Target PDFs:")
    for i, pdf in enumerate(config.TARGET_PDFS, 1):
        print(f"  {i}. {pdf}")
    print(f"\nChunking Parameters:")
    print(f"  • Chunk Size:    {config.CHUNK_SIZE} tokens")
    print(f"  • Chunk Overlap: {config.CHUNK_OVERLAP} tokens\n")

    print("Step 1: Extracting and cleaning text from PDFs...")
    documents = load_all_target_pdfs()
    print(f"Loaded {len(documents)} page documents.\n")

    print("Step 2: Splitting documents using SentenceSplitter...")
    nodes = create_chunks_from_documents(documents)
    stats = get_chunking_statistics(nodes)

    print("\n" + "-" * 70)
    print(f"Total Chunks Generated: {stats['total_chunks']}")
    print(f"Average Chunk Size:     {stats['avg_chars']} chars (~{stats['avg_chars'] // 4} tokens)")
    print(f"Min / Max Chunk Size:   {stats['min_chars']} / {stats['max_chars']} chars")
    print(f"Total Estimated Volume: {stats['approx_total_tokens']} tokens")
    print("-" * 70)

    summary_rows = []
    for doc_title, count in stats["doc_counts"].items():
        summary_rows.append([doc_title, count, f"~{count * int(stats['avg_chars'])} chars"])
    print(tabulate(summary_rows, headers=["Document Title", "Total Chunks", "Est. Content Size"], tablefmt="github"))

    print("\n" + "=" * 70)
    print("                DETAILED CHUNK-BY-CHUNK PREVIEW")
    print("=" * 70)

    for i, node in enumerate(nodes, 1):
        meta = node.metadata
        preview_text = node.text[:220].replace("\n", " ").strip()
        print(f"\n[Chunk #{i:02d}] ----------------------------------------------------")
        print(f"  Doc:        {meta.get('doc_title')}")
        print(f"  Source:     {meta.get('file_name')} (Page {meta.get('page_number')})")
        print(f"  Category:   {meta.get('category')}")
        print(f"  Length:     {meta.get('char_count')} characters (~{meta.get('approx_tokens')} tokens)")
        print(f"  Snippet:    \"{preview_text}...\"")

    if save_json:
        export_path = config.BASE_DIR / "chunks_preview.json"
        export_data = [
            {
                "chunk_id": f"chunk_{i:03d}",
                "doc_title": node.metadata.get("doc_title"),
                "file_name": node.metadata.get("file_name"),
                "page_number": node.metadata.get("page_number"),
                "category": node.metadata.get("category"),
                "char_count": node.metadata.get("char_count"),
                "approx_tokens": node.metadata.get("approx_tokens"),
                "content": node.text
            }
            for i, node in enumerate(nodes, 1)
        ]
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Full chunks exported for inspection to: {export_path}")

    print("=" * 70 + "\n")
    return nodes


if __name__ == "__main__":
    preview_all_chunks()
