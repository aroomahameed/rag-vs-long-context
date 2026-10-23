import pathlib
import tiktoken
import re

RAW = pathlib.Path("data/raw")
enc = tiktoken.get_encoding("cl100k_base")


def ntok(text):
    """Approximate token count. The API reports exact counts later."""
    return len(enc.encode(text, disallowed_special=()))


def read(rel):
    return (RAW / rel).read_text(encoding="utf-8", errors="ignore")


def build_manifests(small_budget=80_000):
    files = sorted(p.relative_to(RAW).as_posix() for p in RAW.rglob("*.md"))
    small, used = [], 0
    for rel in files:
        if not rel.startswith("pydantic/"):
            continue
        t = ntok(read(rel))
        if used + t > small_budget:
            break
        small.append(rel)
        used += t
    pathlib.Path("data/small.txt").write_text("\n".join(small))
    pathlib.Path("data/large.txt").write_text("\n".join(files))
    large_tokens = sum(ntok(read(r)) for r in files)
    print(f"small: {len(small)} files, {used:,} tokens")
    print(f"large: {len(files)} files, {large_tokens:,} tokens")


def load_corpus(name):
    names = pathlib.Path(f"data/{name}.txt").read_text().splitlines()
    return [{"source": n, "text": read(n)} for n in names if n]

def chunk_fixed(doc, size=500, overlap=50):
    """Split a document into windows of `size` tokens that overlap by `overlap`."""
    toks = enc.encode(doc["text"], disallowed_special=())
    step = size - overlap
    return [
        {"id": f'{doc["source"]}#{i}', "source": doc["source"],
         "text": enc.decode(toks[start:start + size])}
        for i, start in enumerate(range(0, len(toks), step))
    ]

def chunk_by_heading(doc, max_tokens=500):
    """Split on Markdown headings; prefix each chunk with its file and section path."""
    sections, path, buf = [], [], []
    in_code = False

    def flush():
        text = "".join(buf).strip()
        if text:
            sections.append((" > ".join(path), text))
        buf.clear()

    for line in doc["text"].splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code = not in_code  # '# comments' inside code are not headings
        m = None if in_code else re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[:level - 1] + [m.group(2).strip()]
        buf.append(line)
    flush()

    chunks = []
    for j, (section, text) in enumerate(sections):
        header = f"[{doc['source']}] {section}\n\n"
        pieces = chunk_fixed({"source": doc["source"], "text": text}, size=max_tokens, overlap=50)
        for k, piece in enumerate(pieces):  # long sections still get split
            chunks.append({"id": f"{doc['source']}#s{j}.{k}", "source": doc["source"],
                           "text": header + piece["text"]})
    return chunks


if __name__ == "__main__":
    build_manifests()
