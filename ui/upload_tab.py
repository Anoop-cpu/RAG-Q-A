import streamlit as st
from ingestion.pdf_loader import extract_pdf_with_images
from ingestion.image_extractor import extract_image_file
from ingestion.convert_markdown import to_markdown
from storage.s3_handler import upload_document, upload_image
from rag.embedder import build_combined_index, build_index
from utils.helper import make_doc_id

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif"]


def _handle_s3_upload(md_text, images, doc_id, metadata):
    """Upload document text and all images to S3, swallowing non-fatal errors."""
    try:
        success = upload_document(content=md_text, doc_id=doc_id, metadata=metadata)
        if not success:
            st.warning("⚠️ S3 document upload failed — continuing without cloud backup.")
        for img in images:
            upload_image(img.base64_data, doc_id, img.index)
    except EnvironmentError as e:
        st.error(f"❌ Configuration error: {e}")
        return False
    except ValueError as e:
        st.error(
            f"❌ Encryption key is invalid: {e}\n\n"
            "Generate a valid key with:\n"
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"`\n\n"
            "Then set it as `ENCRYPTION_KEY` in your `.env` file."
        )
        return False
    return True


def render_upload_tab():
    """Render the PDF + image upload tab UI."""
    st.markdown("Upload a **PDF** (text + embedded images) or a standalone **image file**.")

    file = st.file_uploader(
        "Choose a file",
        type=["pdf"] + SUPPORTED_IMAGE_TYPES,
    )

    if not file:
        return

    is_pdf = file.name.lower().endswith(".pdf")
    is_image = file.name.lower().rsplit(".", 1)[-1] in SUPPORTED_IMAGE_TYPES

    if st.button("Process File", key="process_file"):

        # ── PDF path ──────────────────────────────────────────────────────────
        if is_pdf:
            with st.spinner("Extracting text and images from PDF..."):
                raw_text, images = extract_pdf_with_images(file)
                # raw_text already contains --- Page N --- markers

            with st.spinner("Converting text to markdown..."):
                md_text = to_markdown(raw_text)

            doc_id = make_doc_id(file.name)

            if images:
                st.info(f"🖼️ Found {len(images)} image(s) in the PDF — included in Q&A.")

            with st.spinner("Uploading to S3..."):
                ok = _handle_s3_upload(md_text, images, doc_id, {"source": file.name, "type": "pdf"})
                if not ok:
                    return

            with st.spinner("Building search index..."):
                if images:
                    vectorstore, stored_images = build_combined_index(md_text, images)
                else:
                    vectorstore = build_index(md_text)
                    stored_images = []

            st.session_state.vectorstore = vectorstore
            st.session_state.images = stored_images
            st.session_state.doc_name = file.name
            st.session_state.full_text = md_text    # ← store processed text with page markers
            st.success(f"✅ `{file.name}` processed and ready for Q&A!")

        # ── Standalone image path ─────────────────────────────────────────────
        elif is_image:
            with st.spinner("Loading image..."):
                img = extract_image_file(file)

            doc_id = make_doc_id(file.name)
            st.image(file, caption=f"{file.name} — {img.width}×{img.height}px", use_container_width=True)

            with st.spinner("Uploading to S3..."):
                try:
                    upload_image(img.base64_data, doc_id, 0)
                except (EnvironmentError, ValueError) as e:
                    st.warning(f"⚠️ S3 upload skipped: {e}")

            st.session_state.vectorstore = None
            st.session_state.images = [img]
            st.session_state.doc_name = file.name
            st.session_state.full_text = ""         # ← no text for image-only uploads
            st.success(f"✅ `{file.name}` loaded — ask anything about the image!")