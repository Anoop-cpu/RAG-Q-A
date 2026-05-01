from markdownify import markdownify as md


def to_markdown(raw_text: str) -> str:
    """
    Convert raw extracted text into a clean markdown string.

    Args:
        raw_text: Plain text or HTML-like string from ingestion.

    Returns:
        Cleaned markdown string.
    """
    markdown = md(raw_text, heading_style="ATX", strip=["a"])

    # Collapse excessive blank lines
    lines = markdown.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()