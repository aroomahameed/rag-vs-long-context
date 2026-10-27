import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/runs.csv")
df["score"] = df["grade"].map({"correct": 1.0, "partial": 0.5, "wrong": 0.0})
df["gold_in_context"] = df["gold_in_context"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
df["said_not_found"] = df["said_not_found"].astype(float)
keys = ["corpus", "approach"]
ans = df[df["type"] != "unanswerable"]
unans = df[df["type"] == "unanswerable"]

summary = (
    ans.groupby(keys).agg(accuracy=("score", "mean"),
                          false_refusals=("said_not_found", "mean"),
                          recall5=("recall5", "mean"),
                          gold_in_context=("gold_in_context", "mean"))
    .join(unans.groupby(keys).agg(correct_refusals=("said_not_found", "mean")))
    .join(df.groupby(keys).agg(tokens_in=("tokens_in", "mean"),
                               cost_usd=("cost_usd", "mean"),
                               p50_latency_s=("latency_s", "median"),
                               p95_latency_s=("latency_s", lambda s: s.quantile(0.95))))
)
summary.round(5).to_markdown("results/summary.md")
print(summary.round(5).to_markdown())

# Chart 1: accuracy vs cost (the headline)
fig, ax = plt.subplots(figsize=(7, 5))
for (corpus, approach), r in summary.iterrows():
    ax.scatter(r["cost_usd"], r["accuracy"], marker="o" if corpus == "small" else "s", s=60)
    ax.annotate(f"{approach} ({corpus})", (r["cost_usd"], r["accuracy"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xscale("log")
ax.set(xlabel="Average cost per question (USD, log scale)",
       ylabel="Accuracy (answerable questions)", title="Accuracy vs. cost per question")
fig.savefig("results/accuracy_vs_cost.png", dpi=150, bbox_inches="tight")

# Chart 2: accuracy by question type, large corpus
by_type = df[df["corpus"] == "large"].pivot_table(index="type", columns="approach", values="score")
ax = by_type.plot.bar(figsize=(8, 5), rot=0)
ax.set(ylabel="Accuracy", xlabel="", title="Accuracy by question type (large corpus)")
ax.figure.savefig("results/accuracy_by_type.png", dpi=150, bbox_inches="tight")

# Chart 3: retrieval ablation from Day 4
ret = pd.read_csv("results/retrieval.csv")
abl = ret.groupby(["variant", "corpus"])["recall5"].mean().unstack()
ax = abl.plot(marker="o", figsize=(7, 4))
ax.set(ylabel="Recall@5", xlabel="", title="Retrieval: effect of each tuning step")
ax.figure.savefig("results/retrieval_ablation.png", dpi=150, bbox_inches="tight")

# Failure analysis: why did the best RAG approach get things wrong?
fail = df[(df["approach"] == "tuned_full") & (df["grade"] == "wrong") & (df["type"] != "unanswerable")]
print(f"\ntuned_full wrong answers: {len(fail)}")
print("  retrieval missed the gold file:", int((fail["recall5"] == 0).sum()))
print("  right file retrieved, LLM still wrong:", int((fail["recall5"] == 1).sum()))