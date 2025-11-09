from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
try:
    import faiss  # standard name
except ModuleNotFoundError:
    import faiss_cpu as faiss  # Windows faiss-cpu fallback

from app.utils import iter_texts, chunk_text, save_meta, normalize_ws

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_docs"
STORAGE = ROOT / "storage"
INDEX_DIR = ROOT / "data" / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
STORAGE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def debug_check_index(model: SentenceTransformer, index, texts):
    print("\n[debug] ---- INDEX CHECK ----")
    print(f"[debug] index.ntotal = {index.ntotal}")
    print(f"[debug] index.d      = {index.d}")

    # Brug første chunk som test-query
    test_text = texts[0]
    q_emb = model.encode(
        [test_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")  # shape: (1, dim)

    print(f"[debug] query embedding shape = {q_emb.shape}")

    if q_emb.shape[1] != index.d:
        print(f"[debug] MISMATCH: query dim={q_emb.shape[1]}, index dim={index.d}")
        raise ValueError("Embedding dimension mismatch mellem index og query!")

    # FAISS søgning
    scores, ids = index.search(q_emb, 3)
    print(f"[debug] top-ids: {ids[0].tolist()}")
    print(f"[debug] top-scores: {scores[0].tolist()}")

    # print lille snippet af top-resultat
    top_idx = ids[0][0]
    if 0 <= top_idx < len(texts):
        print("\n[debug] top-resultat snippet:")
        print(texts[top_idx][:200].replace("\n", " "), "...")
    print("[debug] ---- INDEX CHECK DONE ----\n")

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
        batch_size=64,  # mindre batch = mindre RAM, lidt langsommere
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dim = embs.shape[1]

    # 3) FAISS index (cosine ≈ inner product når vektorer er normaliseret)
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))

    # 4) Gem
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "chunks.txt", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    print(f"Saved index with in chunks.txt, with {len(texts)} chunks")

    np.save(INDEX_DIR / "embeddings.npy", embs.astype(np.float32))
    save_meta(INDEX_DIR / "meta.json", {
        "model": MODEL_NAME,
        "count": len(records),
        "dim": int(dim),
        "records": records  # lille korpus? gem direkte; ved stort: gem separat CSV/JSONL
    })
    print(f"OK: {len(records)} chunks, dim={dim}. Index gemt i {INDEX_DIR}/")

    # 5) Self-check: virker index + queries?
    debug_check_index(model, index, texts)

if __name__ == "__main__":
    main()
