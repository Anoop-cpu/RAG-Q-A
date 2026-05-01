"""
Hybrid Indexing Testing Mode
=============================
Compares Dense-only retrieval (FAISS) vs Hybrid retrieval (FAISS + BM25 via RRF)
across all three answer pipelines: RAG, Unrestricted, Synthesized.

Shows:
  - The raw chunks each method retrieved
  - The final answers produced from those chunks
  - Which chunks appear in hybrid but not dense (and vice versa)
  - Retrieval score breakdown for each chunk
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Hybrid Index Testing", page_icon="🔬", layout="wide")

st.title("🔬 Hybrid Indexing Comparison")
st.caption(
    "Compare **Dense-only** (FAISS semantic search) vs **Hybrid** (FAISS + BM25 keyword search) "
    "retrieval — and see how each affects the final answers."
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.get("vectorstore") and not st.session_state.get("images"):
    st.warning("⚠️ No document loaded. Please upload a PDF on the **main page** first.")
    st.stop()

doc_name = st.session_state.get("doc_name", "your document")
st.success(f"📄 Testing against: `{doc_name}`")

# ── Check if hybrid index is available ───────────────────────────────────────
from rag.embedder import HybridIndex

vectorstore = st.session_state.get("vectorstore")
is_hybrid = isinstance(vectorstore, HybridIndex)

if not is_hybrid:
    st.warning(
        "⚠️ The current index is dense-only. "
        "Re-upload your document to build a hybrid index automatically."
    )
    st.stop()

st.divider()

# ── Predefined test questions ─────────────────────────────────────────────────
PREDEFINED = {
    "Exact term / article number": [
        "What does Article 19 say?",
        "What is Article 32?",
        "What are Fundamental Rights?",
    ],
    "Semantic / meaning-based": [
        "What freedoms do citizens have?",
        "How are disputes resolved?",
        "What are the rights of the accused?",
    ],
    "Mixed — keyword + meaning": [
        "What powers does the Supreme Court have?",
        "How does the government ensure equality?",
        "What constitutional protections exist for minorities?",
    ],
}

st.subheader("Select a Test Question")
test_category = st.selectbox("Category", list(PREDEFINED.keys()))
selected_q = st.selectbox("Question", PREDEFINED[test_category])

col1, col2 = st.columns([1, 3])
with col1:
    run_selected = st.button("▶ Run selected", use_container_width=True)
with col2:
    custom_q = st.text_input("Or type your own", placeholder="Ask anything...")
    run_custom = st.button("▶ Run custom", use_container_width=True)

question = None
if run_selected:
    question = selected_q
elif run_custom and custom_q:
    question = custom_q

# ── Run comparison ────────────────────────────────────────────────────────────
if question:
    from rag.retriever import (
        _dense_search, _sparse_search, _reciprocal_rank_fusion,
        _detect_page_query, _extract_page_text,
    )
    from llm.qa_chain import answer_question, answer_unrestricted, answer_synthesized

    st.divider()
    st.markdown(f"**Question:** `{question}`")
    st.divider()

    full_text = st.session_state.get("full_text", "")
    images = st.session_state.get("images") or None
    k = 4

    # ── Page query bypass ─────────────────────────────────────────────────────
    page_number = _detect_page_query(question)
    if page_number and full_text:
        st.info(f"📌 Page query detected — using exact page lookup for page {page_number}. Hybrid comparison not applicable.")
        page_text = _extract_page_text(full_text, page_number)
        context = f"--- Page {page_number} ---\n{page_text}" if page_text else f"[Page {page_number} not found]"
        st.text(context)
        st.stop()

    # ── Retrieve with both methods ────────────────────────────────────────────
    dense_results  = _dense_search(vectorstore.vectorstore, question, k=k)
    sparse_results = _sparse_search(vectorstore.bm25, vectorstore.chunks, question, k=k)
    fused_chunks   = _reciprocal_rank_fusion(dense_results, sparse_results)[:k]

    dense_chunks  = [chunk for chunk, _ in dense_results[:k]]
    hybrid_chunks = fused_chunks

    dense_context  = "\n\n---\n\n".join(dense_chunks)
    hybrid_context = "\n\n---\n\n".join(hybrid_chunks)

    # ── Section 1: Retrieved chunks comparison ────────────────────────────────
    st.subheader("1️⃣ Retrieved Chunks")
    st.caption("What each method fetched from the index for this question.")

    chunk_col1, chunk_col2 = st.columns(2)

    with chunk_col1:
        st.markdown("##### 🔵 Dense Only (FAISS)")
        st.caption("Pure semantic similarity search")
        for i, (chunk, score) in enumerate(dense_results[:k]):
            with st.expander(f"Chunk {i+1} — score {score:.3f}"):
                st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))

    with chunk_col2:
        st.markdown("##### 🟣 Hybrid (FAISS + BM25 via RRF)")
        st.caption("Semantic + keyword search merged with Reciprocal Rank Fusion")
        dense_set  = set(dense_chunks)
        sparse_set = set(c for c, _ in sparse_results[:k])

        for i, chunk in enumerate(hybrid_chunks):
            in_dense  = chunk in dense_set
            in_sparse = chunk in sparse_set
            tag = ""
            if in_dense and in_sparse:
                tag = "🔵🟡 both"
            elif in_dense:
                tag = "🔵 dense only"
            elif in_sparse:
                tag = "🟡 sparse only"

            with st.expander(f"Chunk {i+1} — {tag}"):
                st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))

    # ── Unique chunks analysis ────────────────────────────────────────────────
    st.divider()
    st.subheader("2️⃣ Chunk Overlap Analysis")

    only_in_dense  = [c for c in dense_chunks  if c not in set(hybrid_chunks)]
    only_in_hybrid = [c for c in hybrid_chunks if c not in set(dense_chunks)]
    in_both        = [c for c in dense_chunks  if c in set(hybrid_chunks)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Shared chunks", len(in_both), help="Same chunks retrieved by both methods")
    m2.metric("Dense only", len(only_in_dense), help="Chunks dense found but hybrid did not prioritise")
    m3.metric("Hybrid only", len(only_in_hybrid), help="Chunks hybrid found that dense missed — keyword boost")

    if only_in_hybrid:
        st.markdown("**🟡 Chunks found by Hybrid but NOT by Dense** (keyword boost working):")
        for chunk in only_in_hybrid:
            st.info(chunk[:300] + ("..." if len(chunk) > 300 else ""))

    if only_in_dense:
        with st.expander("🔵 Chunks found by Dense but deprioritised by Hybrid"):
            for chunk in only_in_dense:
                st.text(chunk[:300] + ("..." if len(chunk) > 300 else ""))

    # ── Score breakdown ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("3️⃣ Score Breakdown")
    st.caption("Normalised relevance scores per chunk per method.")

    dense_dict  = {chunk: score for chunk, score in dense_results}
    sparse_dict = {chunk: score for chunk, score in sparse_results}
    all_chunks_set = list(dict.fromkeys(dense_chunks + [c for c, _ in sparse_results[:k]]))

    score_rows = []
    for chunk in all_chunks_set[:8]:
        score_rows.append({
            "Chunk preview": chunk[:80] + "...",
            "Dense score":  round(dense_dict.get(chunk, 0.0), 4),
            "Sparse score": round(sparse_dict.get(chunk, 0.0), 4),
            "In hybrid top-k": "✅" if chunk in set(hybrid_chunks) else "❌",
        })

    st.dataframe(score_rows, use_container_width=True)

    # ── Section 2: Answer comparison ──────────────────────────────────────────
    st.divider()
    st.subheader("4️⃣ Answer Comparison")
    st.caption("How the different retrieved chunks affect the final answers.")

    tab_rag, tab_unr, tab_syn = st.tabs(["📄 RAG", "🌐 Unrestricted", "✨ Synthesized"])

    with tab_rag:
        st.markdown("##### RAG Answer Comparison")
        d_col, h_col = st.columns(2)

        with d_col:
            st.caption("🔵 Dense retrieval")
            ph = st.empty()
            ph.info("⏳ Generating...")
            dense_rag = answer_question(question, dense_context, images=images)
            ph.markdown(dense_rag)

        with h_col:
            st.caption("🟣 Hybrid retrieval")
            ph = st.empty()
            ph.info("⏳ Generating...")
            hybrid_rag = answer_question(question, hybrid_context, images=images)
            ph.markdown(hybrid_rag)

        if dense_rag.strip() == hybrid_rag.strip():
            st.success("✅ Both methods produced the same RAG answer.")
        else:
            st.warning("⚠️ The two methods produced different RAG answers — hybrid retrieval changed the result.")

    with tab_unr:
        st.markdown("##### Unrestricted Answer Comparison")
        d_col, h_col = st.columns(2)

        with d_col:
            st.caption("🔵 Dense context")
            ph = st.empty()
            ph.info("⏳ Generating...")
            dense_unr = answer_unrestricted(question, dense_context, images=images)
            ph.markdown(dense_unr)

        with h_col:
            st.caption("🟣 Hybrid context")
            ph = st.empty()
            ph.info("⏳ Generating...")
            hybrid_unr = answer_unrestricted(question, hybrid_context, images=images)
            ph.markdown(hybrid_unr)

    with tab_syn:
        st.markdown("##### Synthesized Answer Comparison")
        d_col, h_col = st.columns(2)

        with d_col:
            st.caption("🔵 Dense pipeline")
            dense_syn_rag = answer_question(question, dense_context, images=images)
            dense_syn_unr = answer_unrestricted(question, dense_context, images=images)
            ph = st.empty()
            ph.info("⏳ Synthesizing...")
            dense_syn = answer_synthesized(question, dense_syn_rag, dense_syn_unr)
            ph.markdown(dense_syn)

        with h_col:
            st.caption("🟣 Hybrid pipeline")
            ph = st.empty()
            ph.info("⏳ Synthesizing...")
            hybrid_syn = answer_synthesized(question, hybrid_rag, hybrid_unr)
            ph.markdown(hybrid_syn)

    # ── Batch comparison ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("5️⃣ Batch Comparison")
    st.caption("Run multiple questions and compare dense vs hybrid chunk overlap across all of them.")

    if st.button("▶ Run batch comparison"):
        batch_qs = [q for qs in PREDEFINED.values() for q in qs[:1]]
        rows = []
        prog = st.progress(0, text="Running batch...")

        for i, q in enumerate(batch_qs):
            d_res = _dense_search(vectorstore.vectorstore, q, k=k)
            s_res = _sparse_search(vectorstore.bm25, vectorstore.chunks, q, k=k)
            h_chunks = _reciprocal_rank_fusion(d_res, s_res)[:k]
            d_chunks = [c for c, _ in d_res[:k]]

            shared = len(set(d_chunks) & set(h_chunks))
            rows.append({
                "Question": q,
                "Dense chunks": k,
                "Hybrid chunks": k,
                "Shared": shared,
                "Hybrid-only": k - shared,
                "Overlap %": f"{shared/k*100:.0f}%",
            })
            prog.progress((i+1)/len(batch_qs), text=f"{q[:50]}...")

        prog.empty()
        st.dataframe(rows, use_container_width=True)

        avg_overlap = sum(int(r["Shared"]) for r in rows) / len(rows)
        st.metric(
            "Avg chunk overlap",
            f"{avg_overlap:.1f}/{k}",
            help="How many chunks both methods agree on, on average"
        )
