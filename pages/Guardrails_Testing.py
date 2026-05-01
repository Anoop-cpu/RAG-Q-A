"""
Guardrails Testing Mode
=======================
Tests the difference between:
  - With guardrails (RAG): answers only from the document
  - Without guardrails (unrestricted): answers freely from general knowledge
  - Synthesized: merges both, refuses if unrelated

Run predefined test questions or type your own.
Requires a document to be uploaded on the main page first.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Guardrails Testing", page_icon="🛡️", layout="wide")

st.title("🛡️ Guardrails Testing")
st.caption(
    "Compare how each pipeline handles questions that are: "
    "**in the document** · **related but not stated** · **completely unrelated**"
)

# ── Guard: needs a document loaded ───────────────────────────────────────────
if not st.session_state.get("vectorstore") and not st.session_state.get("images"):
    st.warning("⚠️ No document loaded. Please upload a PDF or image on the **main page** first.")
    st.stop()

doc_name = st.session_state.get("doc_name", "your document")
st.success(f"📄 Testing against: `{doc_name}`")
st.divider()

# ── Predefined test questions ─────────────────────────────────────────────────
PREDEFINED_TESTS = {
    "📘 In document": [
        "What is the main purpose of this document?",
        "Summarise what is covered in this document.",
        "What are the key points on page 1?",
    ],
    "🔗 Related but not stated": [
        "What is the historical context behind this document?",
        "How does this compare to similar documents globally?",
        "What are the implications of what this document describes?",
    ],
    "❌ Completely unrelated": [
        "What is the plot of Sharknado?",
        "How do I make pasta carbonara?",
        "Who won the FIFA World Cup in 2022?",
    ],
}

# ── Category selector ─────────────────────────────────────────────────────────
st.subheader("Run Predefined Tests")
category = st.selectbox(
    "Select a test category",
    list(PREDEFINED_TESTS.keys()),
    help="Each category tests a different guardrail scenario.",
)

selected_question = st.selectbox(
    "Select a question",
    PREDEFINED_TESTS[category],
)

col_run, col_custom = st.columns([1, 3])
with col_run:
    run_predefined = st.button("▶ Run this test", use_container_width=True)
with col_custom:
    custom_q = st.text_input("Or type your own question", placeholder="Ask anything...")
    run_custom = st.button("▶ Run custom", use_container_width=True)

question = None
if run_predefined:
    question = selected_question
elif run_custom and custom_q:
    question = custom_q

# ── Run the test ──────────────────────────────────────────────────────────────
if question:
    from llm.qa_chain import answer_question, answer_unrestricted, answer_synthesized
    from rag.retriever import retrieve_context

    st.divider()
    st.markdown(f"**Question:** {question}")
    st.divider()

    context = ""
    if st.session_state.get("vectorstore"):
        context = retrieve_context(
            question,
            st.session_state.vectorstore,
            full_text=st.session_state.get("full_text", ""),
        )

    images = st.session_state.get("images") or None

    # ── Three-column comparison ───────────────────────────────────────────────
    rag_col, unr_col, syn_col = st.columns(3)

    with rag_col:
        st.markdown("#### 📄 With Guardrails (RAG)")
        st.caption("Answers ONLY from the document. Refuses if not found.")
        ph = st.empty()
        ph.info("⏳ Generating...")
        rag_answer = answer_question(question, context, images=images)
        ph.markdown(rag_answer)

    with unr_col:
        st.markdown("#### 🌐 Without Guardrails")
        st.caption("Answers freely from general knowledge. No restrictions.")
        ph = st.empty()
        ph.info("⏳ Generating...")
        unrestricted_answer = answer_unrestricted(question, context, images=images)
        ph.markdown(unrestricted_answer)

    with syn_col:
        st.markdown("#### ✨ Synthesized")
        st.caption("Merges both. Refuses only if truly unrelated.")
        ph = st.empty()
        ph.info("⏳ Synthesizing...")
        synthesized_answer = answer_synthesized(question, rag_answer, unrestricted_answer)
        ph.markdown(synthesized_answer)

    # ── Verdict badge ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔍 Guardrail Verdict")

    rag_refused = any(p in rag_answer.lower() for p in [
        "couldn't find", "not in the document", "not provided", "no information"
    ])
    syn_refused = any(p in synthesized_answer.lower() for p in [
        "not related", "please ask", "unrelated"
    ])

    v1, v2, v3 = st.columns(3)
    with v1:
        if rag_refused:
            st.error("🚫 RAG refused — not in document")
        else:
            st.success("✅ RAG answered from document")
    with v2:
        st.success("✅ Unrestricted always answers")
    with v3:
        if syn_refused:
            st.error("🚫 Synthesized refused — unrelated question")
        else:
            st.success("✅ Synthesized provided an answer")

    # ── Retrieved context ─────────────────────────────────────────────────────
    if context:
        with st.expander("📎 Retrieved context used by RAG"):
            st.text(context)

# ── Batch test all categories ─────────────────────────────────────────────────
st.divider()
st.subheader("Batch Test — All Categories")
st.caption("Runs one question from each category and shows a summary table.")

if st.button("▶ Run batch test", use_container_width=False):
    from llm.qa_chain import answer_question, answer_unrestricted, answer_synthesized
    from rag.retriever import retrieve_context

    results = []
    batch_questions = {
        cat: qs[0] for cat, qs in PREDEFINED_TESTS.items()
    }

    progress = st.progress(0, text="Running batch test...")

    for i, (category_name, q) in enumerate(batch_questions.items()):
        context = ""
        if st.session_state.get("vectorstore"):
            context = retrieve_context(
                q,
                st.session_state.vectorstore,
                full_text=st.session_state.get("full_text", ""),
            )

        rag_ans = answer_question(q, context)
        unr_ans = answer_unrestricted(q, context)
        syn_ans = answer_synthesized(q, rag_ans, unr_ans)

        rag_refused = any(p in rag_ans.lower() for p in [
            "couldn't find", "not in the document", "not provided"
        ])
        syn_refused = any(p in syn_ans.lower() for p in [
            "not related", "please ask", "unrelated"
        ])

        results.append({
            "Category": category_name,
            "Question": q,
            "RAG": "🚫 Refused" if rag_refused else "✅ Answered",
            "Unrestricted": "✅ Answered",
            "Synthesized": "🚫 Refused" if syn_refused else "✅ Answered",
        })

        progress.progress((i + 1) / len(batch_questions), text=f"Tested: {q[:50]}...")

    progress.empty()

    st.markdown("#### Results")
    st.dataframe(results, use_container_width=True)

    # Summary counts
    total = len(results)
    rag_answered = sum(1 for r in results if r["RAG"] == "✅ Answered")
    syn_answered = sum(1 for r in results if r["Synthesized"] == "✅ Answered")

    m1, m2, m3 = st.columns(3)
    m1.metric("RAG answered", f"{rag_answered}/{total}")
    m2.metric("Unrestricted answered", f"{total}/{total}")
    m3.metric("Synthesized answered", f"{syn_answered}/{total}")
