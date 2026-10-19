import pathlib, shutil, subprocess

REPOS = {
    "pydantic": ("https://github.com/pydantic/pydantic", "docs"),
    "fastapi": ("https://github.com/fastapi/fastapi", "docs/en/docs"),
    "sqlmodel": ("https://github.com/fastapi/sqlmodel", "docs"),
    "typer": ("https://github.com/fastapi/typer", "docs"),
}
REPO_DIR = pathlib.Path("data/repos")
RAW = pathlib.Path("data/raw")

for name, (url, sub) in REPOS.items():
    repo = REPO_DIR / name
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
    for md in (repo / sub).rglob("*.md"):
        dest = RAW / name / md.relative_to(repo / sub)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(md, dest)
    print(name, len(list((RAW / name).rglob("*.md"))), "files")