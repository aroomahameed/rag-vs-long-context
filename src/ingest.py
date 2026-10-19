import pathlib
import tiktoken

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


if __name__ == "__main__":
    build_manifests()
    