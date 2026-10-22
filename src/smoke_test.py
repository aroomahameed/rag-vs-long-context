import json, sys
from src.approaches import long_context, naive_rag

corpus = sys.argv[1] if len(sys.argv) > 1 else "small"
qs = [json.loads(l) for l in open("data/eval_questions.jsonl", encoding="utf-8")][:5]
for q in qs:
    print("\nQ:", q["question"], "\nGOLD:", q["gold_answer"])
    for mod in (long_context, naive_rag):
        out = mod.answer(q["question"], corpus)
        print(f"  [{mod.name}] {out['text'][:200]!r}")
        print(f"     {out['tokens_in']:,} tokens in, ${out['cost_usd']:.4f}, {out['latency_s']}s")