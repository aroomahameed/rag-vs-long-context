import os, random
from src.ingest import load_corpus, ntok
from src.llm import ANSWER_PROMPT, call_llm, format_docs

name = "long_context"
BUDGET = int(os.environ.get("CONTEXT_BUDGET", 150_000))
_built = {}


def build_context(corpus):
    if corpus not in _built:
        docs = load_corpus(corpus)
        random.Random(42).shuffle(docs)
        kept, used = [], 0
        for d in docs:
            t = ntok(d["text"]) + 20  # +20 for the <doc> wrapper
            if used + t <= BUDGET:
                kept.append(d)
                used += t
        _built[corpus] = (format_docs(kept), [d["source"] for d in kept], len(docs) - len(kept))
    return _built[corpus]


def answer(question, corpus):
    context, included, n_cut = build_context(corpus)
    out = call_llm(ANSWER_PROMPT.format(context=context, question=question))
    out.update(sources=[], included=included, docs_cut=n_cut)
    return out