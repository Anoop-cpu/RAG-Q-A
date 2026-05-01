import re
import numpy as np
from langchain_community.vectorstores import FAISS


def _detect_page_query(query: str) -> int | None:
    """Detect if the query is asking about a specific page number."""
    patterns = [r'\bpage\s+(\d+)\b', r'\bp\.?\s*(\d+)\b']
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_page_text(full_text: str, page_number: int) -> str:
    """Extract text for a specific page using --- Page N --- markers."""
    pattern = rf'--- Page {page_number} ---\n(.*?)(?=\n{{1,2}}--- Page \d+ ---|$)'
    match = re.search(pattern, full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _dense_search(vectorstore: FAISS, query: str, k: int) -> list[tuple[str, float]]:
    """
    Run FAISS similarity search.
    Returns list of (chunk_text, score) — scores are cosine similarities (0-1).
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    # FAISS returns L2 distance — convert to similarity score (lower = better → invert)
    max_score = max(score for _, score in results) if results else 1.0
    return [(doc.page_content, 1 - (score / (max_score + 1e-9))) for doc, score in results]


def _sparse_search(bm25, chunks: list[str], query: str, k: int) -> list[tuple[str, float]]:
    """
    Run BM25 keyword search.
    Returns list of (chunk_text, score) — scores are BM25 relevance scores (0-1 normalised).
    """
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    # Normalise to 0-1
    max_score = scores.max() if scores.max() > 0 else 1.0
    norm_scores = scores / max_score

    # Get top-k indices
    top_indices = np.argsort(norm_scores)[::-1][:k]
    return [(chunks[i], float(norm_scores[i])) for i in top_indices]


def _reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[str]:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion (RRF).

    RRF assigns each chunk a score based on its rank in each result list,
    not its raw score. This makes the two scales (cosine similarity vs BM25)
    directly comparable.

    Formula: RRF(chunk) = dense_weight / (k + rank_dense)
                        + sparse_weight / (k + rank_sparse)

    Args:
        k:             RRF smoothing constant (default 60, standard in literature).
        dense_weight:  Weight for dense results (default 0.6 — semantic search
                       is generally more reliable for open questions).
        sparse_weight: Weight for sparse results (default 0.4 — keyword search
                       is more reliable for exact terms and article numbers).

    Returns:
        Deduplicated list of chunk strings ordered by fused score, best first.
    """
    scores: dict[str, float] = {}

    for rank, (chunk, _) in enumerate(dense_results):
        scores[chunk] = scores.get(chunk, 0.0) + dense_weight / (k + rank + 1)

    for rank, (chunk, _) in enumerate(sparse_results):
        scores[chunk] = scores.get(chunk, 0.0) + sparse_weight / (k + rank + 1)

    return [chunk for chunk, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve_context(
    query: str,
    vectorstore,                 # FAISS or HybridIndex
    full_text: str = "",
    k: int = 4,
) -> str:
    """
    Retrieve the most relevant document chunks for a query.

    Three retrieval strategies depending on what's available:

    1. PAGE QUERY  — exact page lookup from full_text using markers.
                     Bypasses all vector/keyword search.

    2. HYBRID      — if a HybridIndex is passed, runs both dense (FAISS)
                     and sparse (BM25) search, then merges with RRF.

    3. DENSE ONLY  — if a plain FAISS vectorstore is passed, falls back
                     to standard semantic similarity search.

    Args:
        query:       The user's question.
        vectorstore: Either a HybridIndex (embedder.HybridIndex) or plain FAISS.
        full_text:   Full document text with page markers for page queries.
        k:           Number of results to retrieve per search method.

    Returns:
        Concatenated context string for the LLM.
    """
    # ── Strategy 1: exact page lookup ────────────────────────────────────────
    page_number = _detect_page_query(query)
    if page_number and full_text:
        page_text = _extract_page_text(full_text, page_number)
        if page_text:
            return f"--- Page {page_number} ---\n{page_text}"
        return (
            f"[Page {page_number} was not found in the document. "
            f"The document may not have that many pages.]"
        )

    # ── Strategy 2: hybrid search (dense + sparse) ───────────────────────────
    from rag.embedder import HybridIndex
    if isinstance(vectorstore, HybridIndex):
        dense_results  = _dense_search(vectorstore.vectorstore, query, k=k)
        sparse_results = _sparse_search(vectorstore.bm25, vectorstore.chunks, query, k=k)
        fused_chunks   = _reciprocal_rank_fusion(dense_results, sparse_results)
        top_chunks     = fused_chunks[:k]
        return "\n\n---\n\n".join(top_chunks)

    # ── Strategy 3: dense-only fallback ──────────────────────────────────────
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)