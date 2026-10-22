import json, random
from src.ingest import load_corpus, chunk_fixed
from src.llm import call_llm

random.seed(7)

SINGLE = """Below is a passage from the Pydantic documentation (file: {source}).

{text}

Write ONE question a developer might ask that this passage clearly answers.
{style}
The question must make sense without seeing the passage, so mention Pydantic.
Reply with JSON only: {{"question": "...", "gold_answer": "<short, complete answer>"}}"""

STYLES = {
    "single_fact": "Use plain words.",
    "exact_term": "Use the exact name of a function, class, parameter or setting from the passage.",
}

MULTI = """Below are two passages from the Pydantic documentation.

Passage 1 (file: {s1}):
{t1}

Passage 2 (file: {s2}):
{t2}

Write ONE question that needs facts from BOTH passages to answer. Mention Pydantic.
If no sensible question needs both, reply with JSON only: {{"skip": true}}
Otherwise reply with JSON only: {{"question": "...", "gold_answer": "<short, complete answer>"}}"""


def parse(text):
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        return None


chunks = [c for d in load_corpus("small") for c in chunk_fixed(d, size=400, overlap=0)]
chunks = [c for c in chunks if len(c["text"]) > 800]  # skip tiny fragments
rows = []

for qtype, n in [("single_fact", 45), ("exact_term", 20)]:
    for c in random.sample(chunks, n):
        prompt = SINGLE.format(source=c["source"], text=c["text"], style=STYLES[qtype])
        q = parse(call_llm(prompt, max_tokens=300)["text"])
        if q and "question" in q:
            rows.append({"type": qtype, **q, "gold_sources": [c["source"]]})

for _ in range(30):
    a = random.choice(chunks)
    folder = a["source"].rsplit("/", 1)[0]
    others = [c for c in chunks
              if c["source"].rsplit("/", 1)[0] == folder and c["source"] != a["source"]]
    if not others:
        continue
    b = random.choice(others)
    prompt = MULTI.format(s1=a["source"], t1=a["text"], s2=b["source"], t2=b["text"])
    q = parse(call_llm(prompt, max_tokens=300)["text"])
    if q and "question" in q:
        rows.append({"type": "multi_hop", **q, "gold_sources": [a["source"], b["source"]]})

with open("data/draft_questions.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "draft questions")