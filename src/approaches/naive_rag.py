import time
import chromadb
from sentence_transformers import SentenceTransformer
from src.ingest import chunk_fixed, load_corpus
from src.llm import ANSWER_PROMPT, call_llm, format_docs

name = "naive_rag"
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="index")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed(texts):
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def fill_collection(col, chunks, batch=256):
    if col.count() > 0:
        return  # already built
    for i in range(0, len(chunks), batch):
        b = chunks[i:i + batch]
        col.add(ids=[c["id"] for c in b],
                documents=[c["text"] for c in b],
                metadatas=[{"source": c["source"]} for c in b],
                embeddings=embed([c["text"] for c in b]))


def retrieve(question, corpus, k=5):
    col = client.get_or_create_collection(f"naive_{corpus}")
    fill_collection(col, [c for d in load_corpus(corpus) for c in chunk_fixed(d)])
    r = col.query(query_embeddings=embed([QUERY_PREFIX + question]), n_results=k)
    return [{"id": i, "text": t, "source": m["source"]}
            for i, t, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])]


def answer(question, corpus):
    start = time.time()
    chunks = retrieve(question, corpus)
    retrieval_s = time.time() - start
    out = call_llm(ANSWER_PROMPT.format(context=format_docs(chunks), question=question))
    out["sources"] = [c["source"] for c in chunks]
    out["latency_s"] = round(out["latency_s"] + retrieval_s, 2)  # total = retrieval + LLM
    return out