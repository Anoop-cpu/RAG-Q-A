from dataclasses import dataclass
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from rank_bm25 import BM25Okapi

from rag.chunker import chunk_text
from ingestion.image_extractor import ExtractedImage


def _get_embeddings():
    """
    Returns a FastEmbed embeddings instance.
    FastEmbed runs via ONNX — no PyTorch or GPU required.
    Model downloads once (~50MB) on first use.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@dataclass
class HybridIndex:
    """
    Holds both a dense FAISS index and a sparse BM25 index over the same chunks.

    Dense (FAISS)  — semantic similarity, good for meaning-based queries.
    Sparse (BM25)  — keyword frequency, good for exact terms, article numbers,
                     proper nouns, and legal/technical vocabulary.
    """
    vectorstore: FAISS
    bm25: BM25Okapi
    chunks: list[str]     # raw chunk strings, shared by both indexes


def build_index(text: str) -> FAISS:
    """
    Build a dense-only FAISS index. Used when images are absent.
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks produced from the document.")
    return FAISS.from_texts(chunks, _get_embeddings())


def build_hybrid_index(text: str) -> HybridIndex:
    """
    Build a hybrid dense + sparse index over the document text.

    Dense index  → FAISS with FastEmbed embeddings (semantic search)
    Sparse index → BM25Okapi (keyword / term-frequency search)

    Both indexes operate over the same set of chunks so results
    can be merged by the retriever at query time.

    Args:
        text: Full markdown document text with page markers.

    Returns:
        HybridIndex containing both indexes and the raw chunks.
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks produced from the document.")

    # Dense index
    vectorstore = FAISS.from_texts(chunks, _get_embeddings())

    # Sparse index — tokenise by whitespace for BM25
    tokenised = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenised)

    return HybridIndex(vectorstore=vectorstore, bm25=bm25, chunks=chunks)


def build_image_captions(images: list[ExtractedImage]) -> list[str]:
    """Generate searchable text captions from image metadata."""
    captions = []
    for img in images:
        parts = [f"[Image {img.index + 1}]"]
        if img.page_number:
            parts.append(f"Found on page {img.page_number}.")
        parts.append(f"Dimensions: {img.width}x{img.height} pixels.")
        captions.append(" ".join(parts))
    return captions


def build_combined_index(text: str, images: list[ExtractedImage]) -> tuple[HybridIndex, list[ExtractedImage]]:
    """
    Build a hybrid index from both document text and image captions.

    Args:
        text:   Full markdown document text.
        images: Extracted images from the document.

    Returns:
        Tuple of (HybridIndex, images list).
    """
    text_chunks = chunk_text(text)
    image_captions = build_image_captions(images)
    all_chunks = text_chunks + image_captions

    if not all_chunks:
        raise ValueError("No content produced from the document.")

    vectorstore = FAISS.from_texts(all_chunks, _get_embeddings())
    tokenised = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenised)

    return HybridIndex(vectorstore=vectorstore, bm25=bm25, chunks=all_chunks), images


def benchmark_index_build(text: str) -> dict:
    """
    Time dense vs hybrid index build on the provided text.
    Called by evaluation/metrics.py — kept here so it lives with the indexing code.

    Returns:
        Dict with build times, chunk count, and overhead percentage.
    """
    import time
    from rag.chunker import chunk_text

    chunks = chunk_text(text)
    n_chunks = len(chunks)

    t0 = time.perf_counter()
    build_index(text)
    t_dense = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    build_hybrid_index(text)
    t_hybrid = round(time.perf_counter() - t0, 2)

    return {
        "n_chunks":       n_chunks,
        "dense_build_s":  t_dense,
        "hybrid_build_s": t_hybrid,
        "overhead_s":     round(t_hybrid - t_dense, 2),
        "overhead_pct":   round((t_hybrid - t_dense) / (t_dense + 1e-9) * 100, 1),
    }
