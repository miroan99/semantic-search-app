from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

from utils import iter_texts, chunk_text, save_meta, normalize_ws

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_docs"
STORAGE = ROOT / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # 1) Indlæs og chunk
    records = []  # [{id, doc_path, chunk_text}]
    for p, text in iter_texts(RAW):
        if not text:
            continue
        for idx, ch in enumerate(chunk_text(text, max_chars=800, overlap=100)):
            records.append({
                "id": f"{p.name}::chunk{idx}",
                "doc_path": str(p),
                "text": ch
            })
    print(f"Loaded {len(records)} chunks from text files")
    if not records:
        raise SystemExit("Ingen tekster fundet i data/raw_docs")

    # 2) Embeddings
    model = SentenceTransformer(MODEL_NAME)
    texts = [r["text"] for r in records]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [r["text"] for r in records]
    print(f"Encoding {len(texts)} chunks...")
    embs = model.encode(
        texts,
        batch_size=8,  # mindre batch = mindre RAM, lidt langsommere
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dim = embs.shape[1]

    # 3) FAISS index (cosine ≈ inner product når vektorer er normaliseret)
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    # 4) Gem
    faiss.write_index(index, str(STORAGE / "faiss.index"))
    np.save(STORAGE / "embeddings.npy", embs.astype(np.float32))
    save_meta(STORAGE / "meta.json", {
        "model": MODEL_NAME,
        "count": len(records),
        "dim": int(dim),
        "records": records  # lille korpus? gem direkte; ved stort: gem separat CSV/JSONL
    })
    print(f"OK: {len(records)} chunks, dim={dim}. Index gemt i storage/")

if __name__ == "__main__":
    main()
