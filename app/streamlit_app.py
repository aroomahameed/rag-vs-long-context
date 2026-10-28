import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # so `src` imports work

import streamlit as st
from src.approaches import long_context, naive_rag, tuned_rag

st.set_page_config(page_title="RAG vs Long Context", layout="wide")
st.title("RAG vs Long Context")
st.caption("Same question, same LLM (Claude Haiku 5.5). Only how the context is gathered changes.")

with st.sidebar:
    st.header("What I measured")
    st.markdown(
        "- **Docs fit in context:** long context 100% accurate, but **137x** more expensive than hybrid RAG\n"
        "- **Docs don't fit:** long context fell to **21%**, hybrid RAG held at **91%**"
    )
    chart = pathlib.Path("results/accuracy_vs_cost.png")
    if chart.exists():
        st.image(str(chart), caption="Accuracy vs. cost per question")

corpus = st.radio(
    "Corpus",
    ["small", "large"],
    horizontal=True,
    format_func=lambda c: "Small (Pydantic docs, fits in context)" if c == "small"
    else "Large (4 libraries, doesn't fit)",
)
c1, c2 = st.columns(2)
include_rerank = c1.checkbox("Include tuned + reranker (slow on CPU, ~30 s)")
include_long = c2.checkbox("Include long context (costs ~$0.06-0.09 per question)")
question = st.text_input("Ask a question about the Pydantic docs")

if question:
    approaches = [
        ("Naive RAG", naive_rag.answer),
        ("Tuned hybrid", lambda q, c: tuned_rag.answer(q, c, rerank=False)),
    ]
    if include_rerank:
        approaches.append(("Tuned + reranker", tuned_rag.answer))
    if include_long:
        approaches.insert(0, ("Long context", long_context.answer))

    for col, (label, fn) in zip(st.columns(len(approaches)), approaches):
        with col:
            st.subheader(label)
            with st.spinner("Thinking..."):
                out = fn(question, corpus)
            st.write(out["text"])
            st.caption(f"{out['tokens_in']:,} tokens in · ${out['cost_usd']:.5f} · {out['latency_s']} s")
            if out.get("sources"):
                st.caption("Retrieved: " + ", ".join(dict.fromkeys(out["sources"])))