from typing import List, Dict, Any
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.node_parser import SentenceSplitter
from tabulate import tabulate

import config
from document_processor import load_all_target_pdfs


def build_sentence_splitter(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> SentenceSplitter:
    """
    Constructs a SentenceSplitter that respects sentence boundaries
    and avoids awkward splits in policy clauses and paragraphs.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        separator=" ",
    )


def create_chunks_from_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> List[BaseNode]:
    """
    Splits documents into nodes/chunks with rich metadata.
    Handles short cover pages by appending title context.
    """
    splitter = build_sentence_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_nodes = splitter.get_nodes_from_documents(documents)

    enriched_nodes: List[BaseNode] = []
    
    # Track chunk index per document
    doc_chunk_counters: Dict[str, int] = {}

    for node in raw_nodes:
        # Avoid zero-length or pure whitespace chunks
        text = node.text.strip()
        if not text:
            continue

        file_name = node.metadata.get("file_name", "unknown")
        doc_chunk_counters[file_name] = doc_chunk_counters.get(file_name, 0) + 1
        chunk_idx = doc_chunk_counters[file_name]

        # Enrich node metadata
        node.metadata["chunk_index"] = chunk_idx
        node.metadata["char_count"] = len(text)
        node.metadata["approx_tokens"] = len(text) // 4

        # Configure which metadata keys are visible to embedding and LLM
        node.excluded_embed_metadata_keys = ["char_count", "approx_tokens", "total_pages"]
        node.excluded_llm_metadata_keys = ["char_count", "approx_tokens", "total_pages"]

        enriched_nodes.append(node)

    return enriched_nodes


def get_chunking_statistics(nodes: List[BaseNode]) -> Dict[str, Any]:
    """
    Calculates detailed statistics about the generated chunks.
    """
    if not nodes:
        return {"total_chunks": 0}

    doc_counts: Dict[str, int] = {}
    char_lengths: List[int] = []

    for n in nodes:
        doc_title = n.metadata.get("doc_title", n.metadata.get("file_name", "Unknown"))
        doc_counts[doc_title] = doc_counts.get(doc_title, 0) + 1
        char_lengths.append(len(n.text))

    return {
        "total_chunks": len(nodes),
        "doc_counts": doc_counts,
        "min_chars": min(char_lengths),
        "max_chars": max(char_lengths),
        "avg_chars": round(sum(char_lengths) / len(char_lengths), 1),
        "approx_total_tokens": sum(char_lengths) // 4,
    }


def print_chunk_statistics(nodes: List[BaseNode]) -> None:
    """
    Prints a formatted summary table of chunk statistics.
    """
    stats = get_chunking_statistics(nodes)
    print("\n" + "=" * 65)
    print("                    CHUNKING STATISTICS SUMMARY")
    print("=" * 65)
    print(f"Total Chunks:           {stats['total_chunks']}")
    print(f"Average Chunk Size:     {stats['avg_chars']} characters (~{stats['avg_chars'] // 4} tokens)")
    print(f"Min / Max Chunk Size:   {stats['min_chars']} / {stats['max_chars']} characters")
    print(f"Total Estimated Tokens: {stats['approx_total_tokens']} tokens")
    print("\nChunks Per Document:")

    table_data = [
        [title, count, f"~{count * int(stats['avg_chars'])} chars"]
        for title, count in stats["doc_counts"].items()
    ]
    print(tabulate(table_data, headers=["Document Title", "Chunk Count", "Estimated Volume"], tablefmt="github"))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    docs = load_all_target_pdfs()
    nodes = create_chunks_from_documents(docs)
    print_chunk_statistics(nodes)

    print("\nSample Chunk Preview (First 2 chunks):")
    for i, node in enumerate(nodes[:2]):
        print(f"\n--- [Chunk #{i+1}] {node.metadata['doc_title']} (Page {node.metadata['page_number']}) ---")
        print(node.text[:250] + "...\n")
