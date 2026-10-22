import json
from src.approaches import naive_rag
from src.ingest import chunk_fixed, load_corpus

for corpus in ["small", "large"]:
    col = naive_rag.client.get_or_create_collection(f"naive_{corpus}")
    n_chunks = sum(len(chunk_fixed(d)) for d in load_corpus(corpus))
    print(f"{corpus}: index has {col.count()} chunks, corpus has {n_chunks}")

qs = [json.loads(l) for l in open("data/eval_questions.jsonl", encoding="utf-8")][:5]
for q in qs:
    print("\nQ:", q["question"][:90])
    print("  gold:", q["gold_sources"])
    for corpus in ["small", "large"]:
        got = [c["source"] for c in naive_rag.retrieve(q["question"], corpus)]
        hit = any(s in q["gold_sources"] for s in got)
        print(f"  {corpus}: {'HIT ' if hit else 'MISS'} {got}")