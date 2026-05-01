import requests
from bs4 import BeautifulSoup


def extract_url(url: str, timeout: int = 10) -> str:
    """
    Scrape and extract readable text from a webpage.

    Args:
        url: The webpage URL to scrape.
        timeout: Request timeout in seconds.

    Returns:
        Cleaned text content from the page.

    Raises:
        ValueError: If the URL is invalid or the request fails.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer article or main content if available
    main = soup.find("article") or soup.find("main") or soup.body

    if main is None:
        return soup.get_text(separator="\n", strip=True)

    return main.get_text(separator="\n", strip=True)