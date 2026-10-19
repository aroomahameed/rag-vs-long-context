import hashlib, json, os, pathlib, time
from dotenv import load_dotenv
import litellm

load_dotenv()
MODEL = os.environ["MODEL"]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", MODEL)
PRICE_IN = float(os.environ["PRICE_IN"])
PRICE_OUT = float(os.environ["PRICE_OUT"])
# Optional higher prices for long prompts (defaults to the normal price)
PRICE_IN_LONG = float(os.environ.get("PRICE_IN_LONG", PRICE_IN))
PRICE_OUT_LONG = float(os.environ.get("PRICE_OUT_LONG", PRICE_OUT))
LONG_THRESHOLD = int(os.environ.get("LONG_THRESHOLD", 10**12))
CACHE = pathlib.Path("cache")
CACHE.mkdir(exist_ok=True)

ANSWER_PROMPT = """You answer questions using ONLY the documents below.
Cite the source file(s) in square brackets, like [pydantic/concepts/models.md].
If the answer is not in the documents, reply exactly: NOT FOUND

<documents>
{context}
</documents>

Question: {question}
Answer:"""


def format_docs(items):
    """Wrap documents or chunks with their source so the model can cite them."""
    return "\n\n".join(f'<doc source="{d["source"]}">\n{d["text"]}\n</doc>' for d in items)


def price(tokens_in, tokens_out):
    """Cost in USD of one call, using the long-prompt prices above the threshold."""
    if tokens_in > LONG_THRESHOLD:
        p_in, p_out = PRICE_IN_LONG, PRICE_OUT_LONG
    else:
        p_in, p_out = PRICE_IN, PRICE_OUT
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def call_llm(prompt, model=MODEL, max_tokens=800):
    key = hashlib.sha256(f"{model}|{max_tokens}|{prompt}".encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    start = time.time()
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        num_retries=3,  # retry short rate limits automatically
    )
    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens
    out = {
        "text": resp.choices[0].message.content or "",
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": price(tokens_in, tokens_out),  # judge calls are priced at answer-model rates; only answer costs are reported
        "latency_s": round(time.time() - start, 2),
    }
    cached.write_text(json.dumps(out))
    return out

