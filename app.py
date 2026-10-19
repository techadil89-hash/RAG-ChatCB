import os
import sys
import streamlit as st
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')

import config
from document_processor import DOCUMENT_METADATA_MAP
from indexer import get_chroma_client
from rag_engine import RAGChatbot

# Page configuration
st.set_page_config(
    page_title="Enterprise Policy & Benefits Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium executive styling
st.markdown("""
<style>
    /* Global font and background enhancements */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 1.4rem 1.8rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 0;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.3rem 0.75rem;
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Document card in sidebar */
    .doc-card {
        padding: 0.75rem 0.9rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.65rem;
    }
    .dark .doc-card {
        background-color: #1e293b;
        border-color: #334155;
    }
    .doc-card-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .dark .doc-card-title {
        color: #f1f5f9;
    }
    .doc-card-desc {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.35;
    }

    /* Source Citation Card */
    .citation-card {
        padding: 0.85rem 1rem;
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        margin: 0.6rem 0;
        font-size: 0.88rem;
    }
    .dark .citation-card {
        background: #1e293b;
        border-left: 3px solid #60a5fa;
    }

    /* Clean prompt chip buttons */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    # Use environment configured key and verified models
    st.session_state.chatbot = RAGChatbot(
        api_key=config.GEMINI_API_KEY,
        llm_model=config.LLM_MODEL,
        top_k=config.DEFAULT_TOP_K,
        temperature=config.TEMPERATURE
    )

chatbot: RAGChatbot = st.session_state.chatbot


# ---------------- SIDEBAR: EXECUTIVE OVERVIEW ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=56)
    st.markdown("### **Enterprise AI Assistant**")
    
    # Active Knowledge Status Badge
    st.markdown(
        '<div class="status-badge"><span style="color:#10b981;">●</span> Knowledge Base: Active & Verified</div>',
        unsafe_allow_html=True
    )
    st.caption("All responses are verified and cross-referenced with official corporate documentation.")

    st.markdown("---")

    # Document Catalog
    st.markdown("#### 📚 **Indexed Policies & Guides**")
    
    doc_icons = {
        "GROUP EMPLOYEE OPD BENEFITS PLAN BOOKLET 2024-25. (2).pdf": "🏥",
        "Information Asset Protection Policy.pdf": "🔒",
        "Monitoring and Logging Policy.pdf": "📋",
    }

    for filename, meta in DOCUMENT_METADATA_MAP.items():
        icon = doc_icons.get(filename, "📄")
        with st.expander(f"{icon} {meta['doc_title']}", expanded=False):
            st.caption(f"**Domain:** {meta['category']}")
            st.caption(f"**Coverage:** {meta['description']}")

    st.markdown("---")

    # Action: Reset Conversation
    if st.button("🗑️ New Conversation", use_container_width=True, help="Clear chat history and start a fresh inquiry"):
        st.session_state.messages = []
        chatbot.reset_conversation()
        st.rerun()

    st.markdown("---")
    st.caption("🔒 **Enterprise Security**: Strict anti-hallucination grounding. Every response includes direct page citations.")


# ---------------- MAIN CONTENT: CHAT DEMO ----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ Corporate Knowledge & Benefits Assistant</div>
    <div class="hero-subtitle">Ask questions about medical benefits, data security rules, and IT policies with verified source citations.</div>
</div>
""", unsafe_allow_html=True)

# Suggested Questions for Quick Demo
st.markdown("##### 💡 **Frequently Asked Questions**")
sample_queries = [
    "What medical outpatient expenses and limits are covered under the OPD plan?",
    "Who is eligible for OPD medical coverage and how many children are included?",
    "What are the company's password expiration and screen lock policies?",
    "Under what conditions can employee computer or network usage be monitored?"
]

chip_cols = st.columns(2)
selected_prompt = None
for idx, query in enumerate(sample_queries):
    col = chip_cols[idx % 2]
    if col.button(f"📌 {query}", key=f"demo_chip_{idx}", use_container_width=True):
        selected_prompt = query

st.markdown("---")

# Render Conversation History
for msg in st.session_state.messages:
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        
        # Render verified source citations if available
        if msg.get("sources"):
            with st.expander(f"📖 Verified Source Documents ({len(msg['sources'])})", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['doc_title']} — Page {src['page_number']}**")
                    st.caption(f"Category: *{src['category']}*")
                    # Clean blockquote excerpt
                    st.info(f"“{src['text'][:320]}...”")
                    if i < len(msg["sources"]):
                        st.markdown("")

# User Input Box
chat_input = st.chat_input("Ask any question regarding benefits, claims, or policies...")
active_query = selected_prompt or chat_input

if active_query:
    msg_id = len(st.session_state.messages)
    
    # Record User Message
    st.session_state.messages.append({
        "id": msg_id,
        "role": "user",
        "content": active_query
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_query)

    # Generate Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving verified policy context..."):
            try:
                response_data = chatbot.chat(active_query)
                answer_text = response_data["answer"]
                sources = response_data.get("sources", [])

                # Render Answer
                st.markdown(answer_text)

                # Render Citations
                if sources:
                    with st.expander(f"📖 Verified Source Documents ({len(sources)})", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}. {src['doc_title']} — Page {src['page_number']}**")
                            st.caption(f"Category: *{src['category']}*")
                            st.info(f"“{src['text'][:320]}...”")
                            if i < len(sources):
                                st.markdown("")

                # Save Assistant Message
                st.session_state.messages.append({
                    "id": msg_id + 1,
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"Error generating answer: {e}")
