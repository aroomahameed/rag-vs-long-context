import json
import pandas as pd
from src.approaches import naive_rag, tuned_rag
from src.eval.metrics import mrr, recall_at_k

VARIANTS = {
    "1 naive": lambda q, c: naive_rag.retrieve(q, c),
    "2 + heading chunks": lambda q, c: tuned_rag.retrieve(q, c, hybrid=False, rerank=False),
    "3 + hybrid": lambda q, c: tuned_rag.retrieve(q, c, rerank=False),
    "4 + reranker": lambda q, c: tuned_rag.retrieve(q, c),
}

qs = [json.loads(l) for l in open("data/eval_questions.jsonl", encoding="utf-8")]
qs = [q for q in qs if q["gold_sources"]]  # skip unanswerable
rows = []
for corpus in ["small", "large"]:
    for variant, fn in VARIANTS.items():
        for q in qs:
            sources = [c["source"] for c in fn(q["question"], corpus)]
            rows.append({"corpus": corpus, "variant": variant, "type": q["type"],
                         "recall5": recall_at_k(sources, q["gold_sources"]),
                         "mrr": mrr(sources, q["gold_sources"])})

df = pd.DataFrame(rows)
df.to_csv("results/retrieval.csv", index=False)
print(df.groupby(["corpus", "variant"])[["recall5", "mrr"]].mean().round(3).to_markdown())
print()
print(df[df.corpus == "large"].pivot_table(index="type", columns="variant", values="recall5").round(2).to_markdown())