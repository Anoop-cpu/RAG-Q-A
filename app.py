import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ui.upload_tab import render_upload_tab
from ui.url_tab import render_url_tab

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "vectorstore": None,
    "images": [],
    "doc_name": None,
    "full_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📄 Document Q&A")
st.caption("Upload a PDF, image, or enter a URL — then ask questions about the content.")
st.info("👈 Use the sidebar to switch to **Guardrails Testing** or **Hybrid Indexing** modes.")

# ── Upload tabs ───────────────────────────────────────────────────────────────
upload_tab, url_tab = st.tabs(["📂 Upload PDF / Image", "🌐 Enter URL"])
with upload_tab:
    render_upload_tab()
with url_tab:
    render_url_tab()

# ── Q&A Section ───────────────────────────────────────────────────────────────
st.divider()

has_content = st.session_state.vectorstore or st.session_state.images

if has_content:
    st.subheader(f"💬 Ask about: `{st.session_state.doc_name}`")

    if st.session_state.images:
        with st.expander(f"🖼️ {len(st.session_state.images)} image(s) loaded"):
            cols = st.columns(min(len(st.session_state.images), 4))
            for i, img in enumerate(st.session_state.images):
                col = cols[i % 4]
                col.markdown(
                    f'<img src="data:{img.media_type};base64,{img.base64_data}" '
                    f'style="width:100%;border-radius:6px;" />',
                    unsafe_allow_html=True,
                )
                label = f"Image {img.index + 1}"
                if img.page_number:
                    label += f" (p.{img.page_number})"
                col.caption(label)

    question = st.text_input("Your question", placeholder="What does this document show?")

    if question:
        from llm.qa_chain import answer_question, answer_unrestricted, answer_synthesized
        from rag.retriever import retrieve_context

        context = ""
        if st.session_state.vectorstore:
            context = retrieve_context(
                question,
                st.session_state.vectorstore,
                full_text=st.session_state.full_text,
            )

        images = st.session_state.images or None

        rag_col, unrestricted_col, synthesis_col = st.columns(3)

        with rag_col:
            st.markdown("#### 📄 RAG Answer")
            st.caption("Grounded strictly in your document")
            rag_ph = st.empty()

        with unrestricted_col:
            st.markdown("#### 🌐 Unrestricted Answer")
            st.caption("Document context + general knowledge")
            unr_ph = st.empty()

        with synthesis_col:
            st.markdown("#### ✨ Synthesized Answer")
            st.caption("Best of both combined")
            syn_ph = st.empty()

        rag_ph.info("⏳ Generating...")
        rag_answer = answer_question(question, context, images=images)
        rag_ph.markdown(rag_answer)

        unr_ph.info("⏳ Generating...")
        unrestricted_answer = answer_unrestricted(question, context, images=images)
        unr_ph.markdown(unrestricted_answer)

        syn_ph.info("⏳ Synthesizing...")
        synthesized_answer = answer_synthesized(question, rag_answer, unrestricted_answer)
        syn_ph.markdown(synthesized_answer)

        if context:
            with st.expander("📎 Retrieved context"):
                st.text(context)
else:
    st.info("Upload a PDF, image, or load a URL above to get started.")