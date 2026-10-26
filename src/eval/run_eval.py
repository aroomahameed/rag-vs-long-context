import json, sys
import pandas as pd
from tqdm import tqdm
from src.approaches import long_context, naive_rag, tuned_rag
from src.eval.judge import judge
from src.eval.metrics import mrr, recall_at_k

APPROACHES = {
    "long_context": long_context.answer,
    "naive_rag": naive_rag.answer,
    "tuned_hybrid": lambda q, c: tuned_rag.answer(q, c, rerank=False),
    "tuned_full": tuned_rag.answer,
}


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    corpora = sys.argv[2].split(",") if len(sys.argv) > 2 else ["small", "large"]
    qs = [json.loads(l) for l in open("data/eval_questions.jsonl", encoding="utf-8")][:limit]

    for corpus in corpora:  # build every index before timing anything
        naive_rag.retrieve("warm up", corpus)
        tuned_rag.retrieve("warm up", corpus)
        long_context.build_context(corpus)

    rows = []
    for corpus in corpora:
        for name, fn in APPROACHES.items():
            for q in tqdm(qs, desc=f"{name} / {corpus}"):
                out = fn(q["question"], corpus)
                verdict = judge(q, out["text"])
                gold, srcs = q["gold_sources"], out["sources"]
                rows.append({
                    "corpus": corpus, "approach": name, "qid": q["id"], "type": q["type"],
                    "grade": verdict["grade"], "judge_reason": verdict.get("reason", ""),
                    "answer": out["text"],
                    "said_not_found": "NOT FOUND" in out["text"].upper(),
                    "recall5": recall_at_k(srcs, gold) if srcs else None,
                    "mrr": mrr(srcs, gold) if srcs else None,
                    "gold_in_context": (all(s in out["included"] for s in gold)
                                        if "included" in out and gold else None),
                    "tokens_in": out["tokens_in"], "tokens_out": out["tokens_out"],
                    "cost_usd": out["cost_usd"], "latency_s": out["latency_s"],
                })

    path = "results/runs.csv" if limit is None else "results/runs_test.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main()