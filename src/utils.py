from pathlib import Path

def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
