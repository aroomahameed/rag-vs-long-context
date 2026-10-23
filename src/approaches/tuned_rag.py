import re, time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.approaches.naive_rag import QUERY_PREFIX, client, embed, fill_collection
from src.ingest import chunk_by_heading, load_corpus
from src.llm import ANSWER_PROMPT, call_llm, format_docs

name = "tuned_rag"
reranker = CrossEncoder("BAAI/bge-reranker-base")
_indexes = {}


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def get_index(corpus):
    if corpus not in _indexes:
        chunks = [c for d in load_corpus(corpus) for c in chunk_by_heading(d)]
        col = client.get_or_create_collection(f"tuned_{corpus}")
        fill_collection(col, chunks)
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        _indexes[corpus] = (chunks, {c["id"]: c for c in chunks}, col, bm25)
    return _indexes[corpus]


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(question, corpus, k=5, hybrid=True, rerank=True, pool=30):
    chunks, by_id, col, bm25 = get_index(corpus)
    vec_ids = col.query(query_embeddings=embed([QUERY_PREFIX + question]), n_results=pool)["ids"][0]
    candidates = vec_ids
    if hybrid:
        scores = bm25.get_scores(tokenize(question))
        top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:pool]
        candidates = rrf([vec_ids, [chunks[i]["id"] for i in top]])
    if rerank:
        scores = reranker.predict([(question, by_id[i]["text"]) for i in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda p: p[0], reverse=True)
        candidates = [i for _, i in ranked]
    return [by_id[i] for i in candidates[:k]]


def answer(question, corpus, hybrid=True, rerank=True):
    start = time.time()
    chunks = retrieve(question, corpus, hybrid=hybrid, rerank=rerank)
    retrieval_s = time.time() - start
    out = call_llm(ANSWER_PROMPT.format(context=format_docs(chunks), question=question))
    out["sources"] = [c["source"] for c in chunks]
    out["latency_s"] = round(out["latency_s"] + retrieval_s, 2)
    return out