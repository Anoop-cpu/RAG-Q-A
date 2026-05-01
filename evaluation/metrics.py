"""
evaluation/metrics.py
=====================
Core evaluation functions used by both:
  - pages/3_Metrics_Dashboard.py  (Streamlit UI)
  - evaluation/run_eval.py        (CLI script)

All functions are pure — they take a vectorstore + full_text and
return plain Python dicts/lists so they can be rendered however the
caller wants.
"""

import time
import re
from dataclasses import dataclass, field


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    question: str
    category: str          # "in_document" | "related" | "unrelated"
    rag_answer: str
    unrestricted_answer: str
    synthesised_answer: str
    rag_refused: bool
    syn_refused: bool
    latency_rag_ms: float
    latency_unr_ms: float
    latency_syn_ms: float


@dataclass
class RetrievalResult:
    question: str
    query_type: str        # "exact_term" | "semantic" | "mixed"
    dense_chunks: list[str]
    hybrid_chunks: list[str]
    dense_scores: list[float]
    sparse_scores: list[float]
    shared_count: int
    hybrid_only_count: int
    dense_only_count: int
    overlap_pct: float
    latency_dense_ms: float
    latency_hybrid_ms: float


@dataclass
class PageLookupResult:
    page_number: int
    returned_text: str
    correct: bool          # True if returned text contains page marker
    latency_ms: float


@dataclass
class LatencyResult:
    strategy: str
    latency_ms: float


# ── Refusal detection ─────────────────────────────────────────────────────────

REFUSAL_PHRASES = [
    "couldn't find", "could not find",
    "not in the document", "not provided",
    "no information", "not mentioned",
    "not found in", "does not contain",
]

SYNTHESIS_REFUSAL_PHRASES = [
    "not related to the uploaded document",
    "not related to the document",
    "please ask something about the document",
    "unrelated",
]

def is_rag_refused(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)

def is_syn_refused(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in SYNTHESIS_REFUSAL_PHRASES)


# ── Question banks ────────────────────────────────────────────────────────────
# 54 total questions — all specific to the Indian Constitution
# Guardrail: 18 per category (in_document / related / unrelated) = 54
# Retrieval: 9 per type (exact_term / semantic / mixed) = 27

GUARDRAIL_QUESTIONS = {
    # Questions whose answers are explicitly stated in the Constitution text.
    # Expected: RAG answers, Synthesised answers.
    "in_document": [
        # Specific article queries — clear factual answers in the text
        "What does Article 19 of the Indian Constitution guarantee?",
        "What rights does Article 21 protect?",
        "What does Article 32 allow citizens to do?",
        "What is the procedure for amending the Constitution under Article 368?",
        "What does Article 14 say about equality before the law?",
        "What does Article 17 abolish?",
        "What is stated in Article 51A about fundamental duties?",
        "What does Part III of the Constitution cover?",
        "What does Article 356 empower the President to do?",
        # Structural / factual queries
        "How many articles does the Indian Constitution originally contain?",
        "What are the three branches of government described in the Constitution?",
        "What does the Directive Principles of State Policy section describe?",
        "What qualifications are required to become the President of India?",
        "How is the Vice President of India elected?",
        "What is the role of the Attorney General of India?",
        "What does Schedule VII of the Constitution contain?",
        "What languages are listed in the Eighth Schedule?",
        "What are the powers of the Parliament to make laws?",
    ],
    # Questions about the same subject matter as the document but not
    # explicitly answered in any single retrieved chunk.
    # Expected: RAG refuses, Synthesised uses general knowledge.
    "related": [
        # Historical context
        "When was the Indian Constitution adopted and what was the process?",
        "Who was the Chairman of the Drafting Committee of the Indian Constitution?",
        "How long did it take to draft the Indian Constitution?",
        "What was the influence of the British Government of India Act 1935 on the Constitution?",
        # Comparative constitutional law
        "How does the Indian Constitution compare to the US Constitution?",
        "What makes the Indian Constitution one of the longest in the world?",
        "How does the Indian federal system differ from the American federal system?",
        "What is the significance of the 42nd Amendment to the Indian Constitution?",
        # Broader implications
        "What are the implications of the Right to Equality for citizens?",
        "How has judicial review shaped constitutional interpretation in India?",
        "What is the basic structure doctrine and how does it relate to this document?",
        "How does the Constitution balance individual rights with national security?",
        "What impact has the Constitution had on social reform in India?",
        "How does the Constitution address economic inequality?",
        "What is the significance of the Preamble in constitutional interpretation?",
        "How has the Supreme Court expanded the scope of Article 21 over time?",
        "What is the relationship between Fundamental Rights and Directive Principles?",
        "How does the Constitution protect minority rights in practice?",
    ],
    # Questions with no connection to constitutional law or India.
    # Expected: RAG refuses, Synthesised refuses.
    "unrelated": [
        # Film / entertainment
        "What is the plot of Sharknado?",
        "Who directed the movie Inception?",
        "What are the names of the Avengers in the Marvel Cinematic Universe?",
        # Food / recipes
        "How do I make pasta carbonara?",
        "What is the recipe for French onion soup?",
        "How do you bake a chocolate lava cake?",
        # Sports
        "Who won the FIFA World Cup in 2022?",
        "What are the rules of cricket's Duckworth-Lewis method?",
        "Who holds the record for the most Grand Slam tennis titles?",
        # Technology unrelated to the project
        "How does the Bitcoin blockchain work?",
        "What is the difference between React and Vue.js?",
        "How do you train a neural network from scratch?",
        # Completely random
        "What is the boiling point of mercury?",
        "How do migratory birds navigate?",
        "What caused the extinction of the dinosaurs?",
        "What is the plot of the novel Crime and Punishment?",
        "How does sourdough bread fermentation work?",
        "What is the capital of New Zealand?",
    ],
}

RETRIEVAL_QUESTIONS = {
    # Queries that rely on exact identifier matching — article numbers,
    # schedule names, part numbers. BM25 should help here.
    "exact_term": [
        "What does Article 19 say?",
        "What is Article 32?",
        "What does Article 21 protect?",
        "What is stated in Article 14?",
        "What does Article 356 allow?",
        "What is Part III of the Constitution about?",
        "What does Schedule VII contain?",
        "What are the provisions of Article 368?",
        "What does Article 51A describe?",
    ],
    # Queries framed in natural language without constitutional identifiers.
    # Dense semantic search should handle these better than BM25.
    "semantic": [
        "What freedoms do citizens have?",
        "How are legal disputes resolved?",
        "What protections exist for individual rights?",
        "How does the government protect citizens from arbitrary arrest?",
        "What rights do accused persons have in a trial?",
        "How does the Constitution ensure press freedom?",
        "What are the duties of citizens toward the nation?",
        "How does the Constitution define the relationship between states and the centre?",
        "What safeguards exist against discrimination?",
    ],
    # Queries that mix a specific identifier with a semantic description.
    # Should stress-test both retrieval methods equally.
    "mixed": [
        "What powers does the Supreme Court have?",
        "How does the government ensure equality?",
        "What constitutional protections exist for minorities?",
        "What does the Constitution say about freedom of religion?",
        "How is the President of India elected and what are their powers?",
        "What does the Constitution say about the right to education?",
        "How does Article 300A protect property rights?",
        "What are the emergency powers granted by Part XVIII?",
        "How does the Constitution define the role of the Prime Minister?",
    ],
}

PAGE_TEST_NUMBERS = [1, 5, 10, 20, 50, 75, 100, 150, 200, 300]


# ── Metric 1: Guardrail accuracy ──────────────────────────────────────────────

def run_guardrail_eval(vectorstore, full_text: str) -> list[GuardrailResult]:
    """
    Run all guardrail questions through all three pipelines.
    Returns one GuardrailResult per question.
    """
    from rag.retriever import retrieve_context
    from llm.qa_chain import answer_question, answer_unrestricted, answer_synthesized

    results = []

    for category, questions in GUARDRAIL_QUESTIONS.items():
        for q in questions:
            context = retrieve_context(q, vectorstore, full_text=full_text)

            t0 = time.perf_counter()
            rag_ans = answer_question(q, context)
            t_rag = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            unr_ans = answer_unrestricted(q, context)
            t_unr = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            syn_ans = answer_synthesized(q, rag_ans, unr_ans)
            t_syn = (time.perf_counter() - t0) * 1000

            results.append(GuardrailResult(
                question=q,
                category=category,
                rag_answer=rag_ans,
                unrestricted_answer=unr_ans,
                synthesised_answer=syn_ans,
                rag_refused=is_rag_refused(rag_ans),
                syn_refused=is_syn_refused(syn_ans),
                latency_rag_ms=round(t_rag, 1),
                latency_unr_ms=round(t_unr, 1),
                latency_syn_ms=round(t_syn, 1),
            ))

    return results


def guardrail_summary(results: list[GuardrailResult]) -> dict:
    """Aggregate guardrail results into a summary dict."""
    summary = {}
    for cat in GUARDRAIL_QUESTIONS:
        cat_results = [r for r in results if r.category == cat]
        n = len(cat_results)
        summary[cat] = {
            "total": n,
            "rag_answered": sum(1 for r in cat_results if not r.rag_refused),
            "rag_refused":  sum(1 for r in cat_results if r.rag_refused),
            "syn_answered": sum(1 for r in cat_results if not r.syn_refused),
            "syn_refused":  sum(1 for r in cat_results if r.syn_refused),
            "avg_rag_ms":   round(sum(r.latency_rag_ms for r in cat_results) / n, 1),
            "avg_syn_ms":   round(sum(r.latency_syn_ms for r in cat_results) / n, 1),
        }
    return summary


# ── Metric 2: Retrieval overlap ───────────────────────────────────────────────

def run_retrieval_eval(vectorstore, full_text: str, k: int = 4) -> list[RetrievalResult]:
    """
    Compare dense-only vs hybrid retrieval on all retrieval questions.
    Returns one RetrievalResult per question.
    """
    from rag.embedder import HybridIndex
    from rag.retriever import _dense_search, _sparse_search, _reciprocal_rank_fusion

    if not isinstance(vectorstore, HybridIndex):
        raise ValueError("Retrieval evaluation requires a HybridIndex. Re-upload the document.")

    results = []

    for qtype, questions in RETRIEVAL_QUESTIONS.items():
        for q in questions:
            # Dense
            t0 = time.perf_counter()
            dense_res = _dense_search(vectorstore.vectorstore, q, k=k)
            t_dense = (time.perf_counter() - t0) * 1000

            # Hybrid
            t0 = time.perf_counter()
            sparse_res  = _sparse_search(vectorstore.bm25, vectorstore.chunks, q, k=k)
            fused       = _reciprocal_rank_fusion(dense_res, sparse_res)[:k]
            t_hybrid = (time.perf_counter() - t0) * 1000

            dense_chunks  = [c for c, _ in dense_res[:k]]
            hybrid_chunks = fused
            dense_set     = set(dense_chunks)
            hybrid_set    = set(hybrid_chunks)

            shared        = len(dense_set & hybrid_set)
            hybrid_only   = len(hybrid_set - dense_set)
            dense_only    = len(dense_set - hybrid_set)

            results.append(RetrievalResult(
                question=q,
                query_type=qtype,
                dense_chunks=dense_chunks,
                hybrid_chunks=hybrid_chunks,
                dense_scores=[round(s, 4) for _, s in dense_res[:k]],
                sparse_scores=[round(s, 4) for _, s in sparse_res[:k]],
                shared_count=shared,
                hybrid_only_count=hybrid_only,
                dense_only_count=dense_only,
                overlap_pct=round(shared / k * 100, 1),
                latency_dense_ms=round(t_dense, 1),
                latency_hybrid_ms=round(t_hybrid + t_dense, 1),  # hybrid runs both
            ))

    return results


def retrieval_summary(results: list[RetrievalResult]) -> dict:
    """Aggregate retrieval results by query type."""
    summary = {}
    for qtype in RETRIEVAL_QUESTIONS:
        qr = [r for r in results if r.query_type == qtype]
        n  = len(qr)
        summary[qtype] = {
            "avg_shared":       round(sum(r.shared_count for r in qr) / n, 2),
            "avg_hybrid_only":  round(sum(r.hybrid_only_count for r in qr) / n, 2),
            "avg_dense_only":   round(sum(r.dense_only_count for r in qr) / n, 2),
            "avg_overlap_pct":  round(sum(r.overlap_pct for r in qr) / n, 1),
            "avg_dense_ms":     round(sum(r.latency_dense_ms for r in qr) / n, 1),
            "avg_hybrid_ms":    round(sum(r.latency_hybrid_ms for r in qr) / n, 1),
        }
    return summary


# ── Metric 3: Page lookup accuracy ───────────────────────────────────────────

def run_page_eval(vectorstore, full_text: str) -> list[PageLookupResult]:
    """
    Test exact page lookup for PAGE_TEST_NUMBERS.
    Correct = returned text contains the expected page marker.
    """
    from rag.retriever import retrieve_context

    results = []
    for page_num in PAGE_TEST_NUMBERS:
        q = f"What is on page {page_num}?"
        t0 = time.perf_counter()
        text = retrieve_context(q, vectorstore, full_text=full_text)
        latency = (time.perf_counter() - t0) * 1000

        correct = f"--- Page {page_num} ---" in text or (
            len(text) > 20 and "not found" not in text.lower()
        )
        results.append(PageLookupResult(
            page_number=page_num,
            returned_text=text[:300],
            correct=correct,
            latency_ms=round(latency, 1),
        ))
    return results


# ── Metric 4: Latency benchmark ───────────────────────────────────────────────

def run_latency_benchmark(vectorstore, full_text: str, n_runs: int = 5) -> list[LatencyResult]:
    """
    Benchmark retrieval latency across all three strategies.
    Uses a fixed query run n_runs times and averages.
    """
    from rag.retriever import retrieve_context, _dense_search, _sparse_search, _reciprocal_rank_fusion
    from rag.embedder import HybridIndex

    query = "What are the fundamental rights of citizens?"
    results = []

    # Page lookup latency
    page_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        retrieve_context("What is on page 1?", vectorstore, full_text=full_text)
        page_times.append((time.perf_counter() - t0) * 1000)
    results.append(LatencyResult("Page Lookup", round(sum(page_times) / n_runs, 1)))

    # Dense-only latency
    dense_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if isinstance(vectorstore, HybridIndex):
            _dense_search(vectorstore.vectorstore, query, k=4)
        else:
            vectorstore.similarity_search(query, k=4)
        dense_times.append((time.perf_counter() - t0) * 1000)
    results.append(LatencyResult("Dense Only (FAISS)", round(sum(dense_times) / n_runs, 1)))

    # Hybrid latency
    if isinstance(vectorstore, HybridIndex):
        hybrid_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            dr = _dense_search(vectorstore.vectorstore, query, k=4)
            sr = _sparse_search(vectorstore.bm25, vectorstore.chunks, query, k=4)
            _reciprocal_rank_fusion(dr, sr)
            hybrid_times.append((time.perf_counter() - t0) * 1000)
        results.append(LatencyResult("Hybrid (FAISS + BM25)", round(sum(hybrid_times) / n_runs, 1)))

    return results


# ── Metric 5: Index build time ────────────────────────────────────────────────

def benchmark_index_build(text: str) -> dict:
    """
    Time dense vs hybrid index build on the provided text.
    Returns dict with build times and chunk counts.
    """
    from rag.embedder import build_index, build_hybrid_index
    from rag.chunker import chunk_text

    chunks = chunk_text(text)
    n_chunks = len(chunks)

    t0 = time.perf_counter()
    build_index(text)
    t_dense = round((time.perf_counter() - t0), 2)

    t0 = time.perf_counter()
    build_hybrid_index(text)
    t_hybrid = round((time.perf_counter() - t0), 2)

    return {
        "n_chunks": n_chunks,
        "dense_build_s": t_dense,
        "hybrid_build_s": t_hybrid,
        "overhead_s": round(t_hybrid - t_dense, 2),
        "overhead_pct": round((t_hybrid - t_dense) / t_dense * 100, 1),
    }