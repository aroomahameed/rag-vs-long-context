import json, sys
import pandas as pd

runs = pd.read_csv("results/runs.csv")
keys = ["qid", "approach", "corpus"]

if sys.argv[1:] == ["score"]:
    cal = pd.read_csv("results/calibration.csv")
    merged = cal.merge(runs[keys + ["grade"]], on=keys)
    agree = (merged["grade"] == merged["my_grade"].str.strip().str.lower()).mean()
    print(f"Judge agrees with you on {agree:.0%} of {len(merged)} answers")
    print(merged.loc[merged["grade"] != merged["my_grade"].str.strip().str.lower(), keys + ["grade", "my_grade"]])
else:
    qs = {q["id"]: q for q in map(json.loads, open("data/eval_questions.jsonl", encoding="utf-8"))}
    s = runs.sample(20, random_state=1)
    pd.DataFrame({
        "qid": s["qid"], "approach": s["approach"], "corpus": s["corpus"],
        "question": [qs[i]["question"] for i in s["qid"]],
        "gold_answer": [qs[i]["gold_answer"] for i in s["qid"]],
        "answer": s["answer"], "my_grade": "",
    }).to_csv("results/calibration.csv", index=False)
    print("Fill in my_grade (correct / partial / wrong) in results/calibration.csv")