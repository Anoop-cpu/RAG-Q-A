import io
import fitz  # pymupdf
from ingestion.image_extractor import ExtractedImage, extract_images_from_pdf


def extract_pdf(file) -> str:
    """
    Extract raw text from a PDF file with page number markers embedded.

    Each page's text is prefixed with a marker like:
        --- Page 1 ---
    so that chunks retain page context and page-based queries can be answered.

    Args:
        file: A file-like object or path string to a PDF.

    Returns:
        Extracted text as a single string with page markers.
    """
    if hasattr(file, "read"):
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
    else:
        doc = fitz.open(file)

    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        if page_text:
            pages_text.append(f"--- Page {page_num} ---\n{page_text}")
        else:
            pages_text.append(f"--- Page {page_num} ---\n[No text content on this page]")

    doc.close()
    return "\n\n".join(pages_text)


def extract_pdf_with_images(file) -> tuple[str, list[ExtractedImage]]:
    """
    Extract both text (with page markers) and embedded images from a PDF.

    Args:
        file: A file-like object or path string to a PDF.

    Returns:
        Tuple of (text_content, list_of_extracted_images).
    """
    if hasattr(file, "read"):
        data = file.read()
    else:
        with open(file, "rb") as f:
            data = f.read()

    text = extract_pdf(io.BytesIO(data))
    images = extract_images_from_pdf(io.BytesIO(data))

    return text, images