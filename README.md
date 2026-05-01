# 📄 Document Q&A

An interactive RAG-based application that lets you upload PDF documents or provide webpage URLs and ask natural language questions about their content.

## Features

- 📂 Upload PDF documents for instant Q&A
- 🌐 Load any public webpage by URL
- ☁️ Cloud storage via Amazon S3
- 🔍 Semantic search using FAISS vector index
- 🤖 Answers grounded in document content via Claude (Anthropic)

---

## Project Structure

```
document-qa/
├── app.py                    # Streamlit entry point
├── ingestion/                # PDF & URL extraction + markdown conversion
├── storage/                  # Amazon S3 upload/download
├── rag/                      # Chunking, embedding, retrieval
├── llm/                      # LLM prompt + API call
├── ui/                       # Streamlit tab components
├── utils/                    # Shared helpers
└── tests/                    # Unit tests
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/document-qa.git
cd document-qa
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys and AWS credentials
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (for Claude) |
| `OPENAI_API_KEY` | Your OpenAI API key (for embeddings) |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (default: `us-east-1`) |
| `AWS_BUCKET_NAME` | S3 bucket name for document storage |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | PyMuPDF |
| Web scraping | BeautifulSoup |
| Markdown | markdownify |
| Cloud storage | Amazon S3 (boto3) |
| Vector index | FAISS + LangChain |
| Embeddings | OpenAI |
| LLM | Claude (Anthropic) |