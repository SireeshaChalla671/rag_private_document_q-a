"""
Private Document Intelligence AI (RAG)
========================================
A local, privacy-preserving Retrieval-Augmented Generation chatbot.

Upload one or more PDFs, they're chunked and embedded locally via Ollama,
stored in a persistent Chroma vector database, and you can chat with them
- with conversational memory and source citations for every answer.

Run with:  streamlit run app.py
"""

import time
import uuid
from pathlib import Path

import streamlit as st

from src.config import (
    AVAILABLE_LLM_MODELS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    UPLOAD_DIR,
)
from src.document_processor import load_and_split
from src.document_registry import clear_all as clear_registry
from src.document_registry import list_documents, register_document, remove_document
from src.rag_chain import build_rag_chain, to_lc_messages
from src.reranker import is_reranker_available
from src.theme import THEME_CSS, render_evidence_card
from src.utils import check_ollama_connection, export_chat_as_markdown, format_bytes
from src.vectorstore import (
    add_documents,
    clear_vectorstore,
    delete_by_source,
    get_all_documents,
    get_vectorstore,
    list_indexed_sources,
)

st.set_page_config(page_title="Private Document Intelligence AI", page_icon="🗂️", layout="wide")
st.html(THEME_CSS)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?}]
if "feedback" not in st.session_state:
    st.session_state.feedback = {"up": 0, "down": 0}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂️ Document Intelligence")
    st.caption("Local, private RAG — nothing leaves your machine.")

    # --- Ollama connectivity ---
    is_up, msg = check_ollama_connection()
    if is_up:
        st.success("Ollama: connected", icon="✅")
    else:
        st.error(f"Ollama: {msg}", icon="🚫")
        st.info("Start it with `ollama serve`, then refresh this page.")

    st.divider()
    st.subheader("⚙️ Model Settings")
    llm_model = st.selectbox("LLM model", AVAILABLE_LLM_MODELS, index=AVAILABLE_LLM_MODELS.index(DEFAULT_LLM_MODEL))
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1,
                             help="0 = deterministic/factual, higher = more creative")
    top_k = st.slider("Chunks retrieved (k)", 1, 10, DEFAULT_TOP_K,
                       help="How many chunks are pulled from the knowledge base per question")
    use_hybrid = st.toggle(
        "Hybrid search (keyword + vector)", value=True,
        help="Combines exact keyword matching (BM25) with semantic vector search. "
             "Helps a lot with exact numbers, codes, or names that pure embeddings can miss."
    )

    reranker_ready = is_reranker_available()
    use_reranker = st.toggle(
        "🎯 Cross-encoder reranking", value=reranker_ready, disabled=not reranker_ready,
        help="Retrieves a wider candidate pool, then uses a cross-encoder model to "
             "re-score and reorder them before answering — the biggest lever for "
             "precision on top of hybrid search."
    )
    if not reranker_ready:
        st.caption("⚠️ Run `pip install sentence-transformers` to enable reranking.")

    with st.expander("Advanced: chunking"):
        chunk_size = st.number_input("Chunk size (chars)", 200, 4000, DEFAULT_CHUNK_SIZE, step=100)
        chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 1000, DEFAULT_CHUNK_OVERLAP, step=50)

    st.divider()
    st.subheader("📁 Data Management")
    uploaded_files = st.file_uploader(
        "Upload document(s)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, and TXT. PDFs retain page-level citations.",
    )

    if uploaded_files and st.button("🔨 Index Document(s)", use_container_width=True, disabled=not is_up):
        with st.spinner("Processing: loading → chunking → embedding..."):
            try:
                saved_paths = []
                for uf in uploaded_files:
                    dest = UPLOAD_DIR / uf.name
                    dest.write_bytes(uf.getbuffer())
                    saved_paths.append(dest)

                splits, page_counts = load_and_split(saved_paths, chunk_size, chunk_overlap)
                vectorstore = get_vectorstore()

                # Replace, don't duplicate: re-uploading a file you've already
                # indexed (e.g. after changing chunk size) previously added a
                # second copy of every chunk, which quietly biased retrieval
                # toward whichever file got indexed twice.
                for uf in uploaded_files:
                    delete_by_source(vectorstore, uf.name)
                add_documents(vectorstore, splits)

                for uf in uploaded_files:
                    register_document(uf.name, uf.size, page_counts.get(uf.name, 0))

                # The chunks are now persisted in Chroma; the raw upload was
                # only ever scratch space for the loader. Removing it keeps
                # temp_uploads/ from growing without bound across sessions.
                for path in saved_paths:
                    path.unlink(missing_ok=True)

                st.success(f"Indexed {len(splits)} chunk(s) from {len(saved_paths)} file(s).")
                time.sleep(0.5)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Indexing failed: {e}")

    # --- Document management: show what's indexed, with per-file details ---
    try:
        vectorstore_preview = get_vectorstore()
        sources = list_indexed_sources(vectorstore_preview)
    except Exception:
        sources = {}
    registry = list_documents()

    FILE_TYPE_ICONS = {"PDF": "📕", "DOCX": "📘", "TXT": "📄"}

    if sources:
        st.caption(f"**📚 Indexed documents** ({sum(sources.values())} chunks total)")
        for name, chunk_count in sources.items():
            meta = registry.get(name, {})
            size_str = format_bytes(meta["size_bytes"]) if "size_bytes" in meta else "—"
            pages = meta.get("num_pages", 0)
            file_type = meta.get("file_type", "FILE")
            pages_str = f"{pages} pages" if pages else file_type
            uploaded_str = meta.get("uploaded_at", "—")
            icon = FILE_TYPE_ICONS.get(file_type, "📄")

            with st.container(border=True):
                st.markdown(f"{icon} **{name}**")
                st.caption(f"{size_str} · {pages_str} · {chunk_count} chunks · uploaded {uploaded_str}")
                if st.button("🗑️ Remove", key=f"remove_{name}", use_container_width=True):
                    delete_by_source(vectorstore_preview, name)
                    remove_document(name)
                    st.rerun()

        if "confirm_clear_all" not in st.session_state:
            st.session_state.confirm_clear_all = False

        if not st.session_state.confirm_clear_all:
            if st.button("🗑️ Remove All Documents", use_container_width=True):
                st.session_state.confirm_clear_all = True
                st.rerun()
        else:
            st.warning("Remove all indexed documents? This can't be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear_all = False
                    st.rerun()
            with c2:
                if st.button("Confirm", type="primary", use_container_width=True):
                    clear_vectorstore()
                    clear_registry()
                    st.session_state.messages = []
                    st.session_state.confirm_clear_all = False
                    st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_export:
        st.download_button(
            "⬇️ Export Chat",
            data=export_chat_as_markdown(st.session_state.messages),
            file_name="chat_transcript.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )

    # --- Lightweight evaluation / feedback panel ---
    total_fb = st.session_state.feedback["up"] + st.session_state.feedback["down"]
    if total_fb > 0:
        pct = 100 * st.session_state.feedback["up"] / total_fb
        st.divider()
        st.subheader("📊 Session Feedback")
        st.metric("Helpful answers", f"{pct:.0f}%", f"{total_fb} rated")

# ---------------------------------------------------------------------------
# Main chat window
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="doc-header">
        <span class="seal">&sect;</span>
        <h1>Private Document Intelligence</h1>
    </div>
    <p class="doc-subtitle">A local, cited record of your documents — every answer traces back to the exact page it came from.</p>
    """
)

if not sources and not st.session_state.messages:
    st.info(
        "👋 **Get started:** upload a PDF, DOCX, or TXT file in the sidebar, "
        "click **Index Document(s)**, then ask a question about it here.",
        icon="📁",
    )

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])})"):
                for rank, s in enumerate(message["sources"], start=1):
                    st.html(
                        render_evidence_card(
                            rank, s["source"], s["location"], s["snippet"], s.get("rerank_score")
                        )
                    )
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    st.session_state.feedback["up"] += 1
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    st.session_state.feedback["down"] += 1
                    st.rerun()

if user_query := st.chat_input("Ask a question about your uploaded document(s)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not is_up:
            st.warning("Ollama isn't running — start it and refresh before chatting.")
        elif not sources:
            st.warning("Please upload and index at least one PDF first.")
        else:
            progress = st.status(
                "Searching your documents and preparing an answer...",
                state="running",
                expanded=False,
            )
            try:
                vectorstore = get_vectorstore()
                all_docs = get_all_documents(vectorstore) if use_hybrid else []
                chain = build_rag_chain(
                    vectorstore, llm_model, temperature, top_k,
                    all_docs=all_docs, use_hybrid=use_hybrid, use_reranker=use_reranker,
                )
                chat_history = to_lc_messages(st.session_state.messages[:-1])

                retrieved_docs: list = []

                def _stream_tokens():
                    """Yield answer tokens as they arrive; capture retrieved
                    context (arrives as one chunk before any answer tokens)
                    into retrieved_docs for the citations panel below."""
                    for chunk in chain.stream({"input": user_query, "chat_history": chat_history}):
                        if "context" in chunk:
                            retrieved_docs.extend(chunk["context"])
                        if "answer" in chunk:
                            yield chunk["answer"]

                answer = st.write_stream(_stream_tokens())

                src_list = [
                    {
                        "source": d.metadata.get("source", "unknown"),
                        # PDFs cite a human page number. DOCX/TXT are cited
                        # honestly at document level, not with a made-up page.
                        "location": (
                            f"Page {d.metadata['page'] + 1}"
                            if isinstance(d.metadata.get("page"), int)
                            else "Document"
                        ),
                        "snippet": d.page_content.strip(),
                        "rerank_score": d.metadata.get("rerank_score"),
                    }
                    for d in retrieved_docs
                ]

                if src_list:
                    with st.expander(f"📚 Sources ({len(src_list)})"):
                        for rank, s in enumerate(src_list, start=1):
                            st.html(
                                render_evidence_card(
                                    rank, s["source"], s["location"], s["snippet"], s["rerank_score"]
                                )
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": src_list}
                )
                progress.update(label="Answer ready", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                progress.update(
                    label="Unable to generate an answer", state="error", expanded=False
                )
                st.error(f"An error occurred while generating the answer: {e}")
