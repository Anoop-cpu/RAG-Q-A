import re
import hashlib
from datetime import datetime


def make_doc_id(source: str) -> str:
    """
    Generate a unique, filesystem-safe document ID from a source name or URL.

    Args:
        source: Filename or URL string.

    Returns:
        A short, unique slug string.
    """
    # Strip protocol and special chars
    slug = re.sub(r"https?://", "", source)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
    slug = slug[:40].strip("_")

    # Append a short hash for uniqueness
    short_hash = hashlib.md5(source.encode()).hexdigest()[:6]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    return f"{slug}_{short_hash}_{timestamp}"


def truncate(text: str, max_chars: int = 300) -> str:
    """
    Truncate a string to a maximum length with an ellipsis.

    Args:
        text: Input string.
        max_chars: Maximum character count.

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."