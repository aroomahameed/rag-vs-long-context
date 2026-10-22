import pathlib, sys

term = " ".join(sys.argv[1:]).lower()
hits = [p for p in pathlib.Path("data/raw").rglob("*.md")
        if term in p.read_text(encoding="utf-8", errors="ignore").lower()]
print(len(hits), "files mention it")
for p in hits[:10]:
    print(" ", p)