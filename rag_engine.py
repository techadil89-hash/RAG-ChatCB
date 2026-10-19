import os
import time
from typing import List, Dict, Any, Generator, Optional, Tuple

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import config

SYSTEM_PROMPT = """You are an expert Enterprise AI Assistant specialized in corporate policies and employee benefits for BCER.
You answer user inquiries accurately based ONLY on the provided context retrieved from the official documents:
1. Group Employee OPD Benefits Plan Booklet (2024-25)
2. EMEA Information Asset Protection Policy
3. Monitoring and Logging Policy

CRITICAL INSTRUCTIONS:
- Base your answers strictly and faithfully on the retrieved context.
- If the context does not contain enough information to answer the question, state: "I could not find that information in the provided corporate policy and benefits documents." Do not fabricate or assume unwritten policies.
- Always cite the exact source document and page number for each key point (e.g., "[Source: Group Employee OPD Benefits Plan Booklet (2024-25), Page 2]").
- Format your response clearly using bullet points, bold text for key terms, and numbered lists for sequential procedures.
- Maintain a helpful, objective, and professional tone.
"""


class RAGChatbot:
    """
    Enterprise RAG Chatbot using ChromaDB vector database and Google Gemini models.
    Supports single-turn queries, multi-turn conversational chat, and streaming responses
    with full source citations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = config.LLM_MODEL,
        embedding_model: str = config.EMBEDDING_MODEL,
        top_k: int = config.DEFAULT_TOP_K,
        temperature: float = config.TEMPERATURE
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.temperature = temperature

        self._embed_model: Optional[GoogleGenAIEmbedding] = None
        self._llm: Optional[GoogleGenAI] = None
        self._index: Optional[VectorStoreIndex] = None
        self._chat_engine: Optional[ContextChatEngine] = None
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    @property
    def has_valid_api_key(self) -> bool:
        """Returns True if an API key is configured and not a placeholder."""
        return bool(self.api_key and not self.api_key.startswith("your_"))

    def get_chroma_collection(self) -> Tuple[chromadb.PersistentClient, Any]:
        """Returns ChromaDB persistent client and the target collection."""
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
        return client, collection

    def is_indexed(self) -> bool:
        """Checks if the ChromaDB collection contains indexed documents."""
        try:
            _, collection = self.get_chroma_collection()
            return collection.count() > 0
        except Exception:
            return False

    def get_indexed_count(self) -> int:
        """Returns total number of chunks currently stored in ChromaDB."""
        try:
            _, collection = self.get_chroma_collection()
            return collection.count()
        except Exception:
            return 0

    def init_models(self) -> None:
        """Initializes Gemini LLM and Embedding models."""
        if not self.has_valid_api_key:
            raise ValueError(
                "Gemini API key is required. Please set GEMINI_API_KEY in .env or provide it in the UI."
            )

        if self._embed_model is None:
            self._embed_model = GoogleGenAIEmbedding(
                model_name=self.embedding_model,
                api_key=self.api_key
            )
            Settings.embed_model = self._embed_model

        if self._llm is None:
            self._llm = GoogleGenAI(
                model=self.llm_model,
                api_key=self.api_key,
                temperature=self.temperature
            )
            Settings.llm = self._llm

    def get_index(self) -> VectorStoreIndex:
        """Loads or builds the VectorStoreIndex connected to ChromaDB."""
        if self._index is not None:
            return self._index

        self.init_models()
        _, collection = self.get_chroma_collection()

        # If collection is empty (e.g. freshly deployed without pre-existing db), auto-index
        if collection.count() == 0:
            import indexer
            self._index = indexer.build_or_update_index()
            return self._index

        vector_store = ChromaVectorStore(chroma_collection=collection)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._embed_model
        )
        return self._index

    def retrieve_context(self, query_text: str) -> List[NodeWithScore]:
        """Retrieves top-k relevant nodes for a user query."""
        index = self.get_index()
        retriever = index.as_retriever(similarity_top_k=self.top_k)
        return retriever.retrieve(query_text)

    def get_chat_engine(self) -> ContextChatEngine:
        """Constructs a conversational chat engine with memory and custom system prompt."""
        if self._chat_engine is not None:
            return self._chat_engine

        index = self.get_index()
        retriever = index.as_retriever(similarity_top_k=self.top_k)

        self._chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=self._llm,
            memory=self.memory,
            system_prompt=SYSTEM_PROMPT
        )
        return self._chat_engine

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Executes a multi-turn chat step. Returns the response string along with
        structured source citations.
        """
        engine = self.get_chat_engine()
        response = engine.chat(user_message)

        # Extract source nodes
        sources = []
        source_nodes = getattr(response, "source_nodes", [])
        for node_with_score in source_nodes:
            node = node_with_score.node
            score = getattr(node_with_score, "score", None)
            sources.append({
                "doc_title": node.metadata.get("doc_title", "Document"),
                "file_name": node.metadata.get("file_name", ""),
                "page_number": node.metadata.get("page_number", "N/A"),
                "category": node.metadata.get("category", ""),
                "score": round(score, 4) if score is not None else None,
                "text": node.text.strip()
            })

        return {
            "answer": str(response),
            "sources": sources
        }

    def stream_chat(self, user_message: str):
        """
        Streaming chat generator. Yields tokens as they are produced by Gemini,
        and provides source citations upon completion.
        """
        engine = self.get_chat_engine()
        response_stream = engine.stream_chat(user_message)

        for token in response_stream.response_gen:
            yield token

        # Store sources on the response object
        sources = []
        source_nodes = getattr(response_stream, "source_nodes", [])
        for node_with_score in source_nodes:
            node = node_with_score.node
            score = getattr(node_with_score, "score", None)
            sources.append({
                "doc_title": node.metadata.get("doc_title", "Document"),
                "file_name": node.metadata.get("file_name", ""),
                "page_number": node.metadata.get("page_number", "N/A"),
                "category": node.metadata.get("category", ""),
                "score": round(score, 4) if score is not None else None,
                "text": node.text.strip()
            })
        self._last_sources = sources

    def get_last_sources(self) -> List[Dict[str, Any]]:
        """Returns the sources associated with the most recent streamed response."""
        return getattr(self, "_last_sources", [])

    def reset_conversation(self) -> None:
        """Clears conversational memory buffer."""
        self.memory.reset()
        if self._chat_engine is not None:
            self._chat_engine.reset()
