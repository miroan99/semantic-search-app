# app/utils.py
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re

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

def save_meta(meta_path: Path, meta):
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def embed_text(text: str | list[str]) -> np.ndarray:
    """Return embedding(s) as np.ndarray (n, dim)."""
    if isinstance(text, str):
        text = [text]
    emb = _model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb


def load_faiss_index():
    """Load FAISS index and chunk metadata."""
    index_path = INDEX_DIR / "docs.index"
    meta_path = INDEX_DIR / "chunks_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, chunks


def save_meta(data: dict, name="index_meta.json"):
    """Save metadata (e.g., model name, dimension)."""
    path = INDEX_DIR / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")


def load_meta(name="index_meta.json") -> dict:
    """Load metadata if available."""
    path = INDEX_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

