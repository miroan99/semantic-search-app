from pathlib import Path
import json
import re

def iter_texts(raw_dir: Path):
    for p in raw_dir.glob("**/*"):
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            yield p, normalize_ws(text)

def normalize_ws(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, max_chars: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks of length <= max_chars.
    overlap < max_chars is required, otherwise we risk infinite loops.
    """
    text = text.strip()
    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    if overlap >= max_chars:
        # Enten raise fejl, eller bare disable overlap
        # raise ValueError("overlap must be smaller than max_chars")
        overlap = 0

    chunks = []
    L = len(text)
    i = 0

    while i < L:
        j = min(i + max_chars, L)
        chunks.append(text[i:j])

        if j == L:
            # Vi er nået til enden → stop
            break

        # Flyt frem, med overlap
        i = j - overlap

    return chunks

def save_meta(meta_path: Path, meta):
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
