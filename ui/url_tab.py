import streamlit as st
from ingestion.web_loader import extract_url
from ingestion.convert_markdown import to_markdown
from storage.s3_handler import upload_document
from rag.embedder import build_index
from utils.helper import make_doc_id


def render_url_tab():
    """Render the URL input tab UI."""
    url = st.text_input("Enter a webpage URL", placeholder="https://example.com/article")

    if url and st.button("Load URL", key="load_url"):
        with st.spinner("Fetching webpage..."):
            try:
                raw_text = extract_url(url)
            except ValueError as e:
                st.error(f"❌ {e}")
                return

        with st.spinner("Converting to markdown..."):
            md_text = to_markdown(raw_text)

        doc_id = make_doc_id(url)

        with st.spinner("Uploading to S3..."):
            try:
                success = upload_document(
                    content=md_text,
                    doc_id=doc_id,
                    metadata={"source": url, "type": "url"},
                )
                if not success:
                    st.warning("⚠️ S3 upload failed — continuing without cloud backup.")
            except (EnvironmentError, ValueError) as e:
                st.warning(f"⚠️ S3 upload skipped: {e}")

        with st.spinner("Building search index..."):
            vectorstore = build_index(md_text)

        st.session_state.vectorstore = vectorstore
        st.session_state.images = []      # fix: always reset images when URL loaded
        st.session_state.doc_name = url
        st.success(f"✅ URL loaded and ready for Q&A!")