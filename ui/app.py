"""
ui/app.py

Single unified chat interface.
Agent automatically decides which tool to use:
  - PayPal API Tool   → for API actions
  - RAG Tool          → for knowledge/document questions  
  - System Search     → for capability queries
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="PayPal AI Agent", page_icon="💳", layout="wide")

st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #003087, #009cde);
        padding: 20px 30px; border-radius: 12px;
        color: white; margin-bottom: 20px;
    }
    .tool-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 12px;
        font-weight: 600; margin-bottom: 6px;
    }
    .badge-api    { background:#e8f4fd; color:#003087; border:1px solid #009cde; }
    .badge-rag    { background:#e8f9f0; color:#1a6b3c; border:1px solid #28a745; }
    .badge-search { background:#fff3cd; color:#856404; border:1px solid #ffc107; }
    .chunk-box {
        background: rgba(0,156,222,0.08);
        border-left: 4px solid #009cde;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; margin: 6px 0;
        font-size: 13px; color: inherit;
        white-space: pre-wrap; word-wrap: break-word;
    }
    /* Ensure long responses (e.g. last request status) are fully visible with scroll */
    [data-testid="stChatMessage"] div[data-testid="stMarkdown"] {
        max-height: 70vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    .upload-success {
        background: rgba(40,167,69,0.1);
        border: 1px solid #28a745;
        border-radius: 8px; padding: 10px;
        margin: 4px 0; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h2>💳 PayPal AI Agent</h2>
    <p style="margin:0; opacity:0.9;">
        Powered by LLaMA 3.1 70B • Groq • ChromaDB • LangGraph • LangSmith
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hello! I'm your PayPal AI Agent.\n\n"
            "I can help you with **three types of requests** automatically:\n\n"
            "**💳 PayPal Actions** (calls API)\n"
            "- Send an invoice for $50 to john@example.com\n"
            "- List all my invoices\n"
            "- Create a draft invoice for $100 to alice@test.com\n"
            "- Create an order for $75 USD\n"
            "- What was my total sales volume last month?\n"
            "- Is there a dispute open from user_123?\n\n"
            "**📚 Document Questions** (searches uploaded docs)\n"
            "- What are the Widget view guidelines?\n"
            "- What are the Badge view guidelines?\n"
            "- What dimensions must the widget view be?\n"
            "- What should not be placed in widget view pages?\n"
            "- Summarize the PayPal App UI guidelines\n\n"
            "**🔍 System Search**\n"
            "- What tools are available for managing invoices?\n"
            "- What can this system do for payments?\n"
            "- What tools do I have for subscriptions?\n"
            "- What's the status of my last request?\n\n"
            "Just type naturally — I'll figure out what to do!"
        )
    }]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "request_history" not in st.session_state:
    st.session_state.request_history = []

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b5/PayPal.svg", width=120)
    st.markdown("---")

    st.markdown("### 🔧 System Info")
    st.markdown("""
    - **LLM**: LLaMA 3.1 70B (Groq)
    - **Embeddings**: mxbai-embed-large-v1
    - **Vector DB**: ChromaDB
    - **Framework**: LangGraph
    - **Tracing**: LangSmith
    - **Environment**: PayPal Sandbox
    """)

    st.markdown("---")

    # Tool Registry Status
    st.markdown("### 📊 Tool Registry")
    try:
        import chromadb
        from chromadb.config import Settings
        # Resolve Chroma path from project root so it matches where ingest.py wrote (same as cwd when running ingest)
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
        if os.path.isabs(chroma_dir):
            chroma_path = chroma_dir
        else:
            chroma_path = os.path.normpath(os.path.join(_project_root, chroma_dir))
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        col = client.get_collection("paypal_tools")
        st.success(f"✅ {col.count()} tools indexed")
    except Exception:
        st.error("❌ Run `python ingest.py` first!")

    st.markdown("---")

    # ── Document Upload in Sidebar
    st.markdown("### 📚 Knowledge Base")
    st.markdown("Upload docs for the RAG tool to search:")

    uploaded_file = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        key="doc_uploader"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption(f"📄 {uploaded_file.name}")
        with col2:
            if st.button("Ingest", type="primary", use_container_width=True):
                try:
                    from vectorstore.rag_store import ingest_document, extract_text_from_pdf
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name

                    if filename.endswith(".pdf"):
                        progress = st.empty()
                        progress.caption("⏳ Processing PDF…")
                        text = extract_text_from_pdf(file_bytes)
                        progress.caption("⏳ Embedding chunks (this may take 1–2 min for large PDFs)…")
                    else:
                        text = file_bytes.decode("utf-8", errors="ignore")

                    if not text.strip():
                        st.error("No text found in file.")
                    else:
                        result = ingest_document(
                            text=text,
                            filename=filename,
                            doc_type=filename.split(".")[-1]
                        )
                        if filename.endswith(".pdf"):
                            progress.empty()
                        st.success(f"✅ {result['total_chunks']} chunks stored!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    # Show loaded documents
    try:
        from vectorstore.rag_store import list_documents, delete_document
        docs = list_documents()
        if docs:
            st.markdown(f"**{len(docs)} document(s) loaded:**")
            for d in docs:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(
                        f"<div class='upload-success'>📄 <b>{d['filename']}</b><br>"
                        f"<small>{d['total_chunks']} chunks</small></div>",
                        unsafe_allow_html=True
                    )
                with c2:
                    if st.button("🗑️", key=f"del_{d['doc_id']}", help="Delete"):
                        delete_document(d["doc_id"])
                        st.rerun()
        else:
            st.info("No documents yet.\nUpload one above to enable RAG.")
    except Exception:
        st.info("No documents yet.")

    st.markdown("---")

    # How routing works
    st.markdown("### 🔀 Auto-Routing")
    st.markdown("""
    <div style='font-size:12px'>
    The agent automatically detects:<br><br>
    <span class='tool-badge badge-api'>💳 API Action</span><br>
    → Calls PayPal Sandbox<br><br>
    <span class='tool-badge badge-rag'>📚 RAG Tool</span><br>
    → Searches your docs<br><br>
    <span class='tool-badge badge-search'>🔍 System Search</span><br>
    → Searches tool registry
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# ─────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show tool badge if available
        if message.get("tool_used"):
            tool = message["tool_used"]
            if tool == "api_action" or tool == "multi_step":
                badge = "<span class='tool-badge badge-api'>💳 PayPal API</span>"
            elif tool == "knowledge":
                badge = "<span class='tool-badge badge-rag'>📚 RAG Tool</span>"
            elif tool == "system_search":
                badge = "<span class='tool-badge badge-search'>🔍 System Search</span>"
            else:
                badge = ""
            if badge:
                st.markdown(badge, unsafe_allow_html=True)

        st.markdown(message["content"])

        # Show RAG sources/chunks if available
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- 📄 `{src}`")
        if message.get("chunks"):
            with st.expander(f"🔍 {len(message['chunks'])} chunks retrieved"):
                for c in message["chunks"]:
                    st.markdown(
                        f"<div class='chunk-box'>"
                        f"<b>{c['metadata']['filename']}</b> "
                        f"(score: {c['similarity']})<br>"
                        f"{c['text'][:300]}...</div>",
                        unsafe_allow_html=True
                    )

# ─────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────
prompt = st.chat_input("Ask me anything — PayPal actions, document questions, or tool search...")

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

# ─────────────────────────────────────────
# Process Query
# ─────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔀 Routing • 🔍 Retrieving • 🧠 Planning • ⚡ Executing..."):
            try:
                from agent.graph import run_agent_full
                result = run_agent_full(prompt, request_history=st.session_state.request_history)
                response  = result.get("final_response", "No response generated.")
                intent    = result.get("intent", "")
                sources   = result.get("sources", [])
                chunks    = result.get("chunks", [])
                if "request_history" in result:
                    st.session_state.request_history = result["request_history"]
            except Exception as e:
                response = f"❌ Error: {str(e)}"
                intent, sources, chunks = "", [], []

        # Show tool badge
        if intent:
            if intent in ["api_action", "multi_step"]:
                badge = "<span class='tool-badge badge-api'>💳 PayPal API Tool</span>"
            elif intent == "knowledge":
                badge = "<span class='tool-badge badge-rag'>📚 RAG Tool</span>"
            elif intent == "system_search":
                badge = "<span class='tool-badge badge-search'>🔍 System Search Tool</span>"
            else:
                badge = ""
            if badge:
                st.markdown(badge, unsafe_allow_html=True)

        st.markdown(response)

        if sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"- 📄 `{src}`")
        if chunks:
            with st.expander(f"🔍 {len(chunks)} chunks retrieved"):
                for c in chunks:
                    st.markdown(
                        f"<div class='chunk-box'>"
                        f"<b>{c['metadata']['filename']}</b> "
                        f"(score: {c['similarity']})<br>"
                        f"{c['text'][:300]}...</div>",
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tool_used": intent,
        "sources": sources,
        "chunks": chunks
    })