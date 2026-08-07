import datetime
import streamlit as st
from rag import process_urls, generate_answer, get_vectorstore_stats, reset_vectorstore

# ---------------- PAGE CONFIG & STYLING ---------------- #

st.set_page_config(
    page_title="Real Estate RAG Research Hub",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Custom CSS for layout, cards, fonts, and metric badges
st.markdown(
    """
    <style>
    /* Global Styles & Header */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 1.5rem;
    }
    .main-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Q&A Output Styling */
    .answer-box {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #f1f5f9;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .source-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.35rem 0.75rem;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        text-decoration: none;
    }
    .source-badge:hover {
        background: #334155;
        color: #60a5fa;
    }

    /* Sidebar Section Styling */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SESSION STATE INITIALIZATION ---------------- #

if "urls" not in st.session_state:
    st.session_state.urls = [""]

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = []


# ---------------- SIDEBAR: DYNAMIC SOURCE INGESTION ---------------- #

with st.sidebar:
    st.markdown('<div class="sidebar-header">🏢 Article Sources Ingestion</div>', unsafe_allow_html=True)
    st.caption("Add web URLs for market news, reports, or property data to index into Chroma vector database.")

    # Render dynamic URL inputs
    urls_to_remove = []
    updated_urls = []

    for idx, url in enumerate(st.session_state.urls):
        col_input, col_del = st.columns([5, 1])
        with col_input:
            val = st.text_input(
                f"URL #{idx+1}",
                value=url,
                key=f"url_input_{idx}",
                placeholder="https://example.com/real-estate-article",
                label_visibility="collapsed" if idx > 0 else "visible",
            )
            updated_urls.append(val)
        with col_del:
            # Shift down slightly if label is visible on first element
            if idx == 0:
                st.write("")
                st.write("")
            if st.button("🗑️", key=f"del_{idx}", help=f"Remove URL #{idx+1}"):
                urls_to_remove.append(idx)

    # Apply removals if delete button clicked
    if urls_to_remove:
        for index in sorted(urls_to_remove, reverse=True):
            st.session_state.urls.pop(index)
        st.rerun()

    # Update session state URLs with text input modifications
    st.session_state.urls = updated_urls

    # Action Buttons: Add URL & Clear Inputs
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add URL", help="Add another URL field", use_container_width=True):
            st.session_state.urls.append("")
            st.rerun()
    with col_clear:
        if st.button("🧹 Clear All", help="Reset to single empty URL input", use_container_width=True):
            st.session_state.urls = [""]
            st.rerun()

    st.markdown("---")

    # Ingestion Trigger Button
    process_btn = st.button("🚀 Process & Build Index", type="primary", use_container_width=True)

    if process_btn:
        active_urls = [u.strip() for u in st.session_state.urls if u.strip()]

        if not active_urls:
            st.error("⚠️ Please provide at least one valid URL to process.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg, pct):
                progress_bar.progress(pct)
                status_text.markdown(f"⏳ **{msg}**")

            try:
                with st.spinner("Processing documents & updating Chroma DB..."):
                    summary = process_urls(active_urls, progress_callback=update_progress)

                st.session_state.ingested_urls = active_urls
                st.success(f"✅ Successfully indexed {summary['num_chunks']} chunks from {summary['num_urls']} articles!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Ingestion failed: {str(e)}")

    st.markdown("---")

    # Vectorstore Management
    if st.button("🗑️ Reset Vector Store", use_container_width=True, help="Clear Chroma DB index"):
        reset_vectorstore()
        st.session_state.ingested_urls = []
        st.info("Vector store index reset.")
        st.rerun()

    # Sidebar Footer System Info
    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline Specs")
    st.caption("**LLM**: Groq Llama-3.3-70B-Versatile")
    st.caption("**Embeddings**: Alibaba-NLP/gte-base-en-v1.5")
    st.caption("**Vector DB**: ChromaDB")


# ---------------- MAIN CONTENT AREA ---------------- #

# Main Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">🏢 EstatePulse</h1>
        <p class="main-subtitle">AI-Powered Real Estate Analytics & Market Intelligence using RAG (Retrieval-Augmented Generation)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch current vectorstore stats
stats = get_vectorstore_stats()
chunk_count = stats["chunk_count"]
is_ready = chunk_count > 0

# KPI Dashboard Metrics
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.ingested_urls) if st.session_state.ingested_urls else (len(st.session_state.urls) if is_ready else 0)}</div>
            <div class="metric-label">📰 Ingested Articles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{chunk_count}</div>
            <div class="metric-label">🧩 Indexed Vector Chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.2rem; margin-top:0.4rem;">Llama-3.3-70B</div>
            <div class="metric-label">🧠 Groq LLM Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col4:
    status_label = "🟢 Ready & Active" if is_ready else "🟡 Index Empty"
    color = "#22c55e" if is_ready else "#f59e0b"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color}; font-size: 1.25rem; margin-top: 0.3rem;">{status_label}</div>
            <div class="metric-label">⚡ Chroma DB Status</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------- NAVIGATION TABS ---------------- #

tab_qa, tab_sources, tab_diag = st.tabs(["💬 Ask Questions & Research", "📚 Knowledge Base", "⚙️ System Diagnostics"])


# --- TAB 1: Q&A RESEARCH INTERFACE ---
with tab_qa:
    st.subheader("💡 Market Research Assistant")

    if not is_ready:
        st.info("**Getting Started**: Enter article URLs in the sidebar and click ** Process & Build Index** to begin asking questions!")

    # Question Input
    user_query = st.text_input(
        "Ask any question about mortgage rates, fed policies, or housing market trends:",
        placeholder="e.g. What are the key takeaways regarding current 30-year mortgage rates?",
        key="query_input",
    )

    ask_button = st.button("Generate Answer", type="primary")

    if ask_button or (user_query and user_query != st.session_state.get("last_asked_query", "")):
        if not user_query.strip():
            st.warning("Please type a question before submitting.")
        else:
            st.session_state.last_asked_query = user_query
            with st.spinner("Analyzing vector database and generating answer with Groq Llama 3.3..."):
                try:
                    answer, sources = generate_answer(user_query)

                    # Save to Session Q&A History
                    st.session_state.qa_history.insert(
                        0,
                        {
                            "question": user_query,
                            "answer": answer,
                            "sources": sources,
                            "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        },
                    )
                except Exception as err:
                    st.error(f"⚠️ Query Error: {str(err)}")

    # Display Latest Result if available
    if st.session_state.qa_history:
        latest = st.session_state.qa_history[0]
        st.markdown("---")
        st.markdown(f"###  Answer for: *\"{latest['question']}\"*")

        st.markdown(
            f"""
            <div class="answer-box">
                {latest['answer']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if latest.get("sources"):
            st.markdown("#### 🔗 References & Sources:")
            sources_list = [s.strip() for s in latest["sources"].split("\n") if s.strip()]
            for s in sources_list:
                if s.startswith("http"):
                    st.markdown(f'<a href="{s}" target="_blank" class="source-badge">📄 {s} ↗</a>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="source-badge">📌 {s}</span>', unsafe_allow_html=True)
            st.write("")

    # Q&A History Accordion
    if len(st.session_state.qa_history) > 1:
        st.markdown("---")
        with st.expander(f"📜 Previous Q&A Session History ({len(st.session_state.qa_history) - 1})"):
            for item in st.session_state.qa_history[1:]:
                st.markdown(f"**Q ({item['time']}):** {item['question']}")
                st.markdown(f"> {item['answer']}")
                if item["sources"]:
                    st.caption(f"Sources: {item['sources']}")
                st.markdown("---")


# --- TAB 2: KNOWLEDGE BASE & SOURCES ---
with tab_sources:
    st.subheader("📚 Loaded Article Knowledge Base")

    active_sources = st.session_state.ingested_urls if st.session_state.ingested_urls else [u for u in st.session_state.urls if u.strip()]

    if not active_sources or not is_ready:
        st.warning("No articles have been indexed yet. Ingest URLs in the sidebar to populate the knowledge base.")
    else:
        st.success(f"Total {len(active_sources)} articles indexed into Chroma Vector Database ({chunk_count} vector chunks).")

        for idx, src in enumerate(active_sources):
            with st.container():
                col_num, col_url = st.columns([1, 11])
                with col_num:
                    st.markdown(f"### `#{idx+1}`")
                with col_url:
                    st.markdown(f"**URL:** [{src}]({src})")
                    st.caption("Status: Ingested & Chunked | Vector Store: `real_estate` collection")
                st.markdown("---")


# --- TAB 3: SYSTEM DIAGNOSTISTICS & PIPELINE ARCHITECTURE ---
with tab_diag:
    st.subheader("⚙️ System Architecture & RAG Pipeline Diagnostics")

    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        st.markdown("#### 🤖 AI Models Config")
        st.json(
            {
                "LLM Provider": "Groq API",
                "LLM Model": "llama-3.3-70b-versatile",
                "Temperature": 0.9,
                "Max Output Tokens": 1024,
                "Embedding Model": "Alibaba-NLP/gte-base-en-v1.5",
                "Embedding Dimension / Max Seq": "512 sequence limit",
            }
        )

    with col_diag2:
        st.markdown("#### 📦 Vector Database Config")
        st.json(
            {
                "Vector Store": "Chroma DB",
                "Collection Name": stats["collection_name"],
                "Total Chunks": stats["chunk_count"],
                "Chunk Size": 500,
                "Chunk Overlap": 100
            }
        )

