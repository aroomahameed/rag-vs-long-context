import json
from src.llm import JUDGE_MODEL, call_llm

JUDGE_PROMPT = """You are grading an answer from a documentation Q&A system.

Question: {question}
Gold answer: {gold}
System answer: {answer}

Grades:
- correct: has the key facts of the gold answer and nothing that contradicts it. Extra correct detail is fine.
- partial: some key facts are right, but some are missing or wrong.
- wrong: key facts are wrong or missing, or it says NOT FOUND although a gold answer exists.
Special rule: if the gold answer is NOT FOUND, the system answer is correct only if it says
the answer is not in the documents. Any other answer is wrong.

Reply with JSON only: {{"reason": "<one sentence>", "grade": "correct" | "partial" | "wrong"}}"""


def judge(q, answer_text):
    prompt = JUDGE_PROMPT.format(question=q["question"], gold=q["gold_answer"], answer=answer_text)
    out = call_llm(prompt, model=JUDGE_MODEL, max_tokens=400)
    text = out["text"]
    try:
        verdict = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        verdict = {"reason": "unparseable judge output", "grade": "wrong"}
    if verdict.get("grade") not in ("correct", "partial", "wrong"):
        verdict["grade"] = "wrong"
    return verdict