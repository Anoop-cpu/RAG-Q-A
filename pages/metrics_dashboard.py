"""
Metrics Dashboard
=================
Runs all evaluation metrics live against the currently loaded document,
OR loads previously saved results from evaluation/results/eval_results.json.

Shows:
  1. Guardrail accuracy — answered/refused per pipeline per category
  2. Retrieval overlap  — dense vs hybrid chunk agreement by query type
  3. Page lookup        — accuracy and latency per page tested
  4. Latency benchmark  — ms per retrieval strategy
  5. Index build time   — dense vs hybrid build overhead
"""

import streamlit as st
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Metrics Dashboard")
st.caption("Evaluate retrieval quality, guardrail accuracy, and latency — live or from saved results.")

# ── Sidebar: data source ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    source = st.radio(
        "Select source",
        ["Run live against loaded document", "Load saved results (JSON)"],
    )
    if source == "Load saved results (JSON)":
        results_path = st.text_input(
            "Path to JSON",
            value="evaluation/results/eval_results.json",
        )
    skip_llm = st.checkbox(
        "Skip LLM calls (retrieval + latency only)",
        value=False,
        help="Much faster — skips guardrail evaluation which requires 9 API calls"
    )
    run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

# ── Load or run ───────────────────────────────────────────────────────────────
data = None

if source == "Load saved results (JSON)":
    path = Path(results_path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        st.success(f"Loaded results: `{path.name}` — document: `{data['meta']['document']}`")
    else:
        st.warning(f"No file found at `{results_path}`. Run the CLI first:\n\n"
                   "```bash\npython -m evaluation.run_eval --pdf your_doc.pdf\n```")
        st.stop()

elif run_btn:
    vs = st.session_state.get("vectorstore")
    ft = st.session_state.get("full_text", "")

    if not vs:
        st.error("No document loaded. Upload a PDF on the main page first.")
        st.stop()

    from evaluation.metrics import (
        run_guardrail_eval, guardrail_summary,
        run_retrieval_eval, retrieval_summary,
        run_page_eval, run_latency_benchmark,
        benchmark_index_build,
    )
    from rag.embedder import HybridIndex

    progress = st.progress(0, text="Starting evaluation...")

    # Index build stats
    progress.progress(10, text="Benchmarking index build...")
    build_stats = benchmark_index_build(ft)

    # Retrieval overlap
    progress.progress(25, text="Running retrieval overlap evaluation...")
    try:
        ret_results = run_retrieval_eval(vs, ft)
        ret_sum     = retrieval_summary(ret_results)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Page lookup
    progress.progress(50, text="Testing page lookup accuracy...")
    page_results = run_page_eval(vs, ft)
    correct = sum(1 for r in page_results if r.correct)

    # Latency benchmark
    progress.progress(65, text="Benchmarking retrieval latency...")
    latency_results = run_latency_benchmark(vs, ft)

    # Guardrail eval (LLM)
    grail_results = []
    grail_sum     = {}
    if not skip_llm:
        progress.progress(75, text="Running guardrail evaluation (9 API calls)...")
        grail_results = run_guardrail_eval(vs, ft)
        grail_sum     = guardrail_summary(grail_results)

    progress.progress(100, text="Done!")
    progress.empty()

    # Build data dict matching CLI output format
    data = {
        "meta": {
            "document":    st.session_state.get("doc_name", "Unknown"),
            "page_count":  ft.count("--- Page "),
            "word_count":  len(ft.split()),
            "chunk_count": build_stats["n_chunks"],
            "llm_enabled": not skip_llm,
        },
        "index_build": build_stats,
        "retrieval_overlap": {
            "summary": ret_sum,
            "per_question": [
                {
                    "question":          r.question,
                    "query_type":        r.query_type,
                    "shared":            r.shared_count,
                    "hybrid_only":       r.hybrid_only_count,
                    "dense_only":        r.dense_only_count,
                    "overlap_pct":       r.overlap_pct,
                    "latency_dense_ms":  r.latency_dense_ms,
                    "latency_hybrid_ms": r.latency_hybrid_ms,
                }
                for r in ret_results
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
            "summary": grail_sum,
            "per_question": [
                {
                    "question":       r.question,
                    "category":       r.category,
                    "rag_refused":    r.rag_refused,
                    "syn_refused":    r.syn_refused,
                    "latency_rag_ms": r.latency_rag_ms,
                    "latency_syn_ms": r.latency_syn_ms,
                }
                for r in grail_results
            ],
        },
    }

    # Save for later
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    with open("evaluation/results/eval_results.json", "w") as f:
        json.dump(data, f, indent=2)
    st.success("Evaluation complete — results saved to `evaluation/results/eval_results.json`")

else:
    st.info("Upload a document on the main page, then click **Run Evaluation** in the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# RENDER RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

# ── Document metadata ─────────────────────────────────────────────────────────
meta = data["meta"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Document", meta.get("document", "—")[:20])
m2.metric("Pages",    meta.get("page_count", "—"))
m3.metric("Words",    f"{meta.get('word_count', 0):,}")
m4.metric("Chunks",   meta.get("chunk_count", "—"))

st.divider()

# ── Metric 1: Guardrail accuracy ──────────────────────────────────────────────
st.subheader("1️⃣ Guardrail Accuracy")
st.caption("How each pipeline handles in-document, related, and unrelated questions.")

if data["guardrail"]["summary"]:
    grail = data["guardrail"]["summary"]
    cats  = list(grail.keys())
    labels = {"in_document": "📘 In Document", "related": "🔗 Related", "unrelated": "❌ Unrelated"}

    # Summary table
    rows = []
    for cat in cats:
        s = grail[cat]
        rows.append({
            "Category":          labels.get(cat, cat),
            "n":                 s["total"],
            "RAG Answered":      s["rag_answered"],
            "RAG Refused":       s["rag_refused"],
            "Synth Answered":    s["syn_answered"],
            "Synth Refused":     s["syn_refused"],
            "Avg RAG ms":        s["avg_rag_ms"],
            "Avg Synth ms":      s["avg_syn_ms"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Bar chart: answered counts
    st.markdown("**Pipeline answered vs refused by category**")
    chart_data = {
        "Category": [labels.get(c, c) for c in cats],
        "RAG Answered": [grail[c]["rag_answered"] for c in cats],
        "RAG Refused":  [grail[c]["rag_refused"]  for c in cats],
        "Synth Answered": [grail[c]["syn_answered"] for c in cats],
        "Synth Refused":  [grail[c]["syn_refused"]  for c in cats],
    }

    import pandas as pd
    df = pd.DataFrame(chart_data).set_index("Category")
    st.bar_chart(df[["RAG Answered", "RAG Refused", "Synth Answered", "Synth Refused"]])

    # Verdict scorecard
    st.markdown("**Expected behaviour scorecard**")
    expected = {
        "in_document": {"RAG": "answer", "Synth": "answer"},
        "related":     {"RAG": "refuse", "Synth": "answer"},
        "unrelated":   {"RAG": "refuse", "Synth": "refuse"},
    }
    sc1, sc2, sc3 = st.columns(3)
    for col, cat in zip([sc1, sc2, sc3], cats):
        s   = grail[cat]
        exp = expected.get(cat, {})
        with col:
            st.markdown(f"**{labels.get(cat, cat)}**")
            rag_ok = (exp["RAG"] == "answer" and s["rag_answered"] == s["total"]) or \
                     (exp["RAG"] == "refuse"  and s["rag_refused"]  == s["total"])
            syn_ok = (exp["Synth"] == "answer" and s["syn_answered"] == s["total"]) or \
                     (exp["Synth"] == "refuse"  and s["syn_refused"]  == s["total"])
            st.markdown(f"RAG:   {'✅ Correct' if rag_ok else '⚠️ Unexpected'}")
            st.markdown(f"Synth: {'✅ Correct' if syn_ok else '⚠️ Unexpected'}")
else:
    st.info("Guardrail evaluation was skipped (LLM calls disabled). Re-run without `--no-llm`.")

st.divider()

# ── Metric 2: Retrieval overlap ───────────────────────────────────────────────
st.subheader("2️⃣ Hybrid vs Dense Retrieval Overlap")
st.caption("How many chunks each method agrees on, by query type. Hybrid-only chunks = BM25 keyword boost.")

ret = data["retrieval_overlap"]
if ret["summary"]:
    import pandas as pd

    qtypes  = list(ret["summary"].keys())
    qlabels = {"exact_term": "Exact Term", "semantic": "Semantic", "mixed": "Mixed"}

    # Summary table
    sum_rows = []
    for qt in qtypes:
        s = ret["summary"][qt]
        sum_rows.append({
            "Query Type":      qlabels.get(qt, qt),
            "Avg Shared":      s["avg_shared"],
            "Avg Hybrid-only": s["avg_hybrid_only"],
            "Avg Dense-only":  s["avg_dense_only"],
            "Avg Overlap %":   f"{s['avg_overlap_pct']}%",
            "Dense ms":        s["avg_dense_ms"],
            "Hybrid ms":       s["avg_hybrid_ms"],
            "Overhead ms":     round(s["avg_hybrid_ms"] - s["avg_dense_ms"], 1),
        })
    st.dataframe(sum_rows, use_container_width=True, hide_index=True)

    # Stacked bar: chunk composition
    st.markdown("**Chunk composition per query type (avg across questions)**")
    bar_df = pd.DataFrame({
        "Query Type":  [qlabels.get(qt, qt) for qt in qtypes],
        "Shared":      [ret["summary"][qt]["avg_shared"]      for qt in qtypes],
        "Hybrid-only": [ret["summary"][qt]["avg_hybrid_only"] for qt in qtypes],
        "Dense-only":  [ret["summary"][qt]["avg_dense_only"]  for qt in qtypes],
    }).set_index("Query Type")
    st.bar_chart(bar_df)

    # Per-question detail
    with st.expander("📋 Per-question retrieval detail"):
        pq = pd.DataFrame(ret["per_question"])[
            ["question", "query_type", "shared", "hybrid_only", "dense_only",
             "overlap_pct", "latency_dense_ms", "latency_hybrid_ms"]
        ]
        pq.columns = ["Question", "Type", "Shared", "Hybrid-only",
                      "Dense-only", "Overlap %", "Dense ms", "Hybrid ms"]
        st.dataframe(pq, use_container_width=True, hide_index=True)

st.divider()

# ── Metric 3: Page lookup accuracy ───────────────────────────────────────────
st.subheader("3️⃣ Page Lookup Accuracy")
st.caption("Deterministic regex extraction — should be 100% accurate.")

pl = data["page_lookup"]
pages = pl["per_page"]

p1, p2, p3 = st.columns(3)
p1.metric("Accuracy", f"{pl['accuracy_pct']}%")
p2.metric("Correct",  sum(1 for p in pages if p["correct"]))
p3.metric("Avg Latency", f"{round(sum(p['latency_ms'] for p in pages)/len(pages), 1)} ms")

page_rows = []
for p in pages:
    page_rows.append({
        "Page":       p["page"],
        "Correct":    "✅" if p["correct"] else "❌",
        "Latency ms": p["latency_ms"],
        "Preview":    p["preview"][:80] + "..." if len(p.get("preview","")) > 80 else p.get("preview",""),
    })
st.dataframe(page_rows, use_container_width=True, hide_index=True)

st.divider()

# ── Metric 4: Latency benchmark ───────────────────────────────────────────────
st.subheader("4️⃣ Retrieval Latency Benchmark")
st.caption("Average over 5 runs. Hybrid overhead vs dense-only.")

lat = data["latency"]
if lat:
    import pandas as pd

    lat_df = pd.DataFrame(lat).set_index("strategy")
    st.bar_chart(lat_df["avg_ms"])

    lat_rows = []
    base_ms  = lat[0]["avg_ms"] if lat else 1
    for l in lat:
        lat_rows.append({
            "Strategy":    l["strategy"],
            "Avg ms":      l["avg_ms"],
            "vs Page Lookup": f"+{round(l['avg_ms'] - base_ms, 1)} ms",
        })
    st.dataframe(lat_rows, use_container_width=True, hide_index=True)

st.divider()

# ── Metric 5: Index build time ────────────────────────────────────────────────
st.subheader("5️⃣ Index Build Time")
st.caption("Time to build dense-only vs hybrid index. Overhead should be minimal.")

ib = data["index_build"]
b1, b2, b3, b4 = st.columns(4)
b1.metric("Chunks",        ib["n_chunks"])
b2.metric("Dense build",   f"{ib['dense_build_s']}s")
b3.metric("Hybrid build",  f"{ib['hybrid_build_s']}s")
b4.metric("Overhead",      f"{ib['overhead_pct']}%")

import pandas as pd
build_df = pd.DataFrame({
    "Index":   ["Dense Only", "Hybrid"],
    "Time (s)": [ib["dense_build_s"], ib["hybrid_build_s"]],
}).set_index("Index")
st.bar_chart(build_df)

st.divider()

# ── Download results ──────────────────────────────────────────────────────────
st.subheader("💾 Export Results")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "⬇ Download JSON",
        data=json.dumps(data, indent=2),
        file_name="eval_results.json",
        mime="application/json",
        use_container_width=True,
    )
with col2:
    # Build a flat CSV for the retrieval per-question data
    import csv, io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "question", "query_type", "shared", "hybrid_only",
        "dense_only", "overlap_pct", "latency_dense_ms", "latency_hybrid_ms"
    ])
    writer.writeheader()
    for row in data["retrieval_overlap"]["per_question"]:
        writer.writerow(row)
    st.download_button(
        "⬇ Download Retrieval CSV",
        data=buf.getvalue(),
        file_name="retrieval_results.csv",
        mime="text/csv",
        use_container_width=True,
    )