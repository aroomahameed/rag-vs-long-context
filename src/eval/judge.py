import json
import re
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


def parse_verdict(text):
    """Read the judge's JSON; if it's malformed, pull the grade out directly."""
    try:
        v = json.loads(text[text.find("{"): text.rfind("}") + 1])
        if v.get("grade") in ("correct", "partial", "wrong"):
            return v
    except ValueError:
        pass
    m = re.search(r'"grade"\s*:\s*"(correct|partial|wrong)"', text)
    if m:
        return {"reason": "(recovered from malformed judge output)", "grade": m.group(1)}
    return None


def judge(q, answer_text):
    prompt = JUDGE_PROMPT.format(question=q["question"], gold=q["gold_answer"], answer=answer_text)
    for max_tokens in (400, 1000):  # a different max_tokens is a new cache entry, so the retry is a real call
        verdict = parse_verdict(call_llm(prompt, model=JUDGE_MODEL, max_tokens=max_tokens)["text"])
        if verdict:
            return verdict
    return {"reason": "unparseable judge output", "grade": "wrong"}