"""
evaluation/run_eval.py
======================
Standalone CLI evaluation runner.

Runs all six metric groups against a PDF document and saves results to
evaluation/results/eval_results.json for inspection or report generation.

Usage:
    # From project root with virtualenv active:
    python -m evaluation.run_eval --pdf path/to/document.pdf

    # With custom output path:
    python -m evaluation.run_eval --pdf path/to/document.pdf --out evaluation/results/my_run.json

    # Skip LLM calls (retrieval metrics only, much faster):
    python -m evaluation.run_eval --pdf path/to/document.pdf --no-llm
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Add project root to path so imports work ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation metrics for the RAG Q&A system.")
    parser.add_argument("--pdf",    required=True,  help="Path to the PDF document to evaluate against")
    parser.add_argument("--out",    default="evaluation/results/eval_results.json", help="Output JSON path")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (retrieval + latency only)")
    parser.add_argument("--pages",  nargs="+", type=int, default=None, help="Page numbers to test (default: 1 5 10 20 50)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RAG System Evaluation")
    print(f"  Document : {pdf_path.name}")
    print(f"  LLM calls: {'disabled' if args.no_llm else 'enabled'}")
    print(f"  Output   : {out_path}")
    print(f"{'='*60}\n")

    from ingestion.pdf_extractor import extract_pdf_with_images
    from ingestion.markdown_converter import to_markdown
    from rag.embedder import build_hybrid_index, benchmark_index_build
    from evaluation.metrics import (
        run_guardrail_eval, guardrail_summary,
        run_retrieval_eval, retrieval_summary,
        run_page_eval, run_latency_benchmark,
    )

    if args.pages:
        import evaluation.metrics as m
        m.PAGE_TEST_NUMBERS = args.pages

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    print("[ 1/6 ] Ingesting document...")
    t0 = time.perf_counter()
    with open(pdf_path, "rb") as f:
        raw_text, images = extract_pdf_with_images(f)
    md_text = to_markdown(raw_text)
    ingest_time = round(time.perf_counter() - t0, 2)
    word_count  = len(md_text.split())
    page_count  = md_text.count("--- Page ")
    print(f"        {page_count} pages, {word_count:,} words  ({ingest_time}s)")

    # ── Step 2: Index build benchmark ─────────────────────────────────────────
    print("[ 2/6 ] Benchmarking index build...")
    build_stats = benchmark_index_build(md_text)
    print(f"        {build_stats['n_chunks']} chunks | "
          f"dense={build_stats['dense_build_s']}s | "
          f"hybrid={build_stats['hybrid_build_s']}s | "
          f"overhead={build_stats['overhead_pct']}%")

    # ── Step 3: Build hybrid index ────────────────────────────────────────────
    print("[ 3/6 ] Building hybrid index for evaluation...")
    t0 = time.perf_counter()
    vectorstore = build_hybrid_index(md_text)
    print(f"        Done ({round(time.perf_counter() - t0, 2)}s)")

    # ── Step 4: Retrieval overlap ─────────────────────────────────────────────
    print("[ 4/6 ] Running retrieval overlap evaluation...")
    retrieval_results = run_retrieval_eval(vectorstore, md_text)
    ret_summary = retrieval_summary(retrieval_results)
    for qtype, stats in ret_summary.items():
        print(f"        [{qtype:12s}] overlap={stats['avg_overlap_pct']}%  "
              f"hybrid_only={stats['avg_hybrid_only']:.1f}  "
              f"dense={stats['avg_dense_ms']}ms  hybrid={stats['avg_hybrid_ms']}ms")

    # ── Step 5: Page lookup accuracy ──────────────────────────────────────────
    print("[ 5/6 ] Testing page lookup accuracy...")
    page_results = run_page_eval(vectorstore, md_text)
    correct = sum(1 for r in page_results if r.correct)
    print(f"        {correct}/{len(page_results)} correct  "
          f"avg={round(sum(r.latency_ms for r in page_results)/len(page_results), 1)}ms")

    # ── Step 6: Latency benchmark ─────────────────────────────────────────────
    print("[ 6/6 ] Benchmarking retrieval latency (5 runs each)...")
    latency_results = run_latency_benchmark(vectorstore, md_text)
    for lr in latency_results:
        print(f"        {lr.strategy:<25s} {lr.latency_ms}ms")

    # ── Optional: LLM guardrail eval ─────────────────────────────────────────
    guardrail_results = []
    grail_summary     = {}
    if not args.no_llm:
        print("\n[ LLM ] Running guardrail evaluation (makes API calls)...")
        print("        This will take 1-3 minutes...")
        guardrail_results = run_guardrail_eval(vectorstore, md_text)
        grail_summary     = guardrail_summary(guardrail_results)
        for cat, stats in grail_summary.items():
            print(f"        [{cat:12s}] RAG answered={stats['rag_answered']}/{stats['total']}  "
                  f"syn refused={stats['syn_refused']}/{stats['total']}")
    else:
        print("\n[ LLM ] Skipped (--no-llm flag set)")

    # ── Compile and save ──────────────────────────────────────────────────────
    output = {
        "meta": {
            "document":    pdf_path.name,
            "timestamp":   datetime.now().isoformat(),
            "page_count":  page_count,
            "word_count":  word_count,
            "chunk_count": build_stats["n_chunks"],
            "llm_enabled": not args.no_llm,
        },
        "index_build": build_stats,
        "retrieval_overlap": {
            "summary": ret_summary,
            "per_question": [
                {
                    "question":        r.question,
                    "query_type":      r.query_type,
                    "shared":          r.shared_count,
                    "hybrid_only":     r.hybrid_only_count,
                    "dense_only":      r.dense_only_count,
                    "overlap_pct":     r.overlap_pct,
                    "latency_dense_ms":  r.latency_dense_ms,
                    "latency_hybrid_ms": r.latency_hybrid_ms,
                }
                for r in retrieval_results
            ],
        },
        "page_lookup": {
            "accuracy_pct": round(correct / len(page_results) * 100, 1),
            "per_page": [
                {
                    "page":       r.page_number,
                    "correct":    r.correct,
                    "latency_ms": r.latency_ms,
                    "preview":    r.returned_text[:150],
                }
                for r in page_results
            ],
        },
        "latency": [
            {"strategy": lr.strategy, "avg_ms": lr.latency_ms}
            for lr in latency_results
        ],
        "guardrail": {
            "summary": grail_summary,
            "per_question": [
                {
                    "question":    r.question,
                    "category":    r.category,
                    "rag_refused": r.rag_refused,
                    "syn_refused": r.syn_refused,
                    "latency_rag_ms": r.latency_rag_ms,
                    "latency_syn_ms": r.latency_syn_ms,
                }
                for r in guardrail_results
            ],
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {out_path}")
    print(f"  View in Streamlit: pages/3_Metrics_Dashboard.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
