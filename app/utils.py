# app/utils.py
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import faiss
except ModuleNotFoundError:
    import faiss_cpu as faiss

import re
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_DIR = DATA / "index"
STORAGE = ROOT / "storage"

# ✅ 1) Embedder (use the same model you used when building index)
_EMBED_MODEL = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_EMBED_MODEL)

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

def save_meta(path: Path, metas: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[saved] {path}")


def embed_text(text: str | list[str]) -> np.ndarray:
    """Return embedding(s) as np.ndarray (n, dim)."""
    if isinstance(text, str):
        text = [text]
    emb = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def load_faiss_index():
    index = faiss.read_index(str(INDEX_DIR / "docs.index"))
    # læs metadata/chunks
    chunks: List[Dict] = []
    with open(STORAGE / "index_meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return index, chunks

def load_meta(name="index_meta.json") -> dict:
    """Load metadata if available."""
    path = INDEX_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

