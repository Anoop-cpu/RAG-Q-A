import base64
import io
from dataclasses import dataclass

import fitz  # pymupdf
from PIL import Image


@dataclass
class ExtractedImage:
    """Represents a single extracted image with metadata."""
    base64_data: str        # base64-encoded image bytes
    media_type: str         # e.g. "image/png"
    page_number: int | None # page it came from (None for standalone uploads)
    index: int              # position among all images extracted
    width: int
    height: int


def extract_images_from_pdf(file) -> list[ExtractedImage]:
    """
    Extract all embedded images from a PDF file.

    Args:
        file: File-like object or path string to a PDF.

    Returns:
        List of ExtractedImage objects, one per embedded image found.
    """
    if hasattr(file, "read"):
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
    else:
        doc = fitz.open(file)

    extracted = []
    index = 0

    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            img_bytes = base_image["image"]
            ext = base_image.get("ext", "png").lower()
            media_type = _ext_to_media_type(ext)

            # Normalise to PNG via Pillow for consistency
            img_bytes, media_type = _normalise_image(img_bytes, media_type)

            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

            # Get dimensions
            try:
                with Image.open(io.BytesIO(img_bytes)) as im:
                    width, height = im.size
            except Exception:
                width, height = 0, 0

            # Skip tiny images (icons, bullets, decorative elements)
            if width < 50 or height < 50:
                continue

            extracted.append(ExtractedImage(
                base64_data=b64,
                media_type=media_type,
                page_number=page_num,
                index=index,
                width=width,
                height=height,
            ))
            index += 1

    doc.close()
    return extracted


def extract_image_file(file) -> ExtractedImage:
    """
    Load a standalone uploaded image file (PNG, JPG, WEBP, GIF).

    Args:
        file: A file-like object from st.file_uploader.

    Returns:
        A single ExtractedImage.
    """
    raw = file.read()
    media_type = _guess_media_type(file.name)
    img_bytes, media_type = _normalise_image(raw, media_type)

    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    with Image.open(io.BytesIO(img_bytes)) as im:
        width, height = im.size

    return ExtractedImage(
        base64_data=b64,
        media_type=media_type,
        page_number=None,
        index=0,
        width=width,
        height=height,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_image(img_bytes: bytes, media_type: str) -> tuple[bytes, str]:
    """Convert any image format to PNG for consistent handling."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            return buf.getvalue(), "image/png"
    except Exception:
        return img_bytes, media_type


def _ext_to_media_type(ext: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
    }.get(ext, "image/png")


def _guess_media_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    return _ext_to_media_type(ext)