import json

rows = [json.loads(l) for l in open("data/reviewed.jsonl", encoding="utf-8") if l.strip()]
with open("data/eval_questions.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows, 1):
        f.write(json.dumps({"id": f"q{i:03d}", **r}, ensure_ascii=False) + "\n")
print(len(rows), "questions")