"""Starter PDF ingestion utilities.

Main objectives:
1. Accept a PDF file path (the FastAPI upload layer can save uploads first).
2. Extract text page by page.
3. Preserve metadata such as page number, file name, and section title if possible.
"""

import fitz
from bs4 import BeautifulSoup
import requests

#pdf extractor
def extract_pdf(file_path):
    doc = fitz.open(file_path)
    return "/n".join(page.get_text() for page in doc)

#webpage extractor
def extract_url(url):
    html = requests.get(url).text
    return BeautifulSoup(html, "html.parser").get_text()