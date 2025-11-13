# app/build_index.py
from pathlib import Path
import json
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

    test_text = texts[0]
    q_emb = model.encode([test_text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    print(f"[debug] query embedding shape = {q_emb.shape}")

    if q_emb.shape[1] != index.d:
        print(f"[debug] MISMATCH: query dim={q_emb.shape[1]}, index dim={index.d}")
        raise ValueError("Embedding dimension mismatch mellem index og query!")

    scores, ids = index.search(q_emb, 3)
    print(f"[debug] top-ids: {ids[0].tolist()}")
    print(f"[debug] top-scores: {scores[0].tolist()}")

    top_idx = ids[0][0]
    if 0 <= top_idx < len(texts):
        print("\n[debug] top-resultat snippet:")
        print(texts[top_idx][:200].replace("\n", " "), "...")
    print("[debug] ---- INDEX CHECK DONE ----\n")

def main():
    # 1) Indlæs og chunk
    records = []  # [{id, doc_path, text}]
    for p, text in iter_texts(RAW):
        if not text:
            continue
        for idx, ch in enumerate(chunk_text(text, max_chars=800, overlap=100)):
            records.append({
                "id": f"{p.name}::chunk{idx}",
                "doc_path": str(p),
                "text": normalize_ws(ch),
                "source": p.name,
                "chunk_id": idx,
            })
    print(f"Loaded {len(records)} chunks from text files")
    if not records:
        raise SystemExit("Ingen tekster fundet i data/raw_docs")

    # 2) Embeddings (én model-init)
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [r["text"] for r in records]
    print(f"Encoding {len(texts)} chunks...")
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    dim = int(embs.shape[1])

    # 3) FAISS index (cosine via IP + normaliserede vektorer)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # 4) Gem index + chunk-metadata i formater som rag_pipeline/utils forventer
    faiss.write_index(index, str(INDEX_DIR / "docs.index"))  # <-- matcher utils.load_faiss_index
    # Gem chunk-liste (korte felter)
    chunks = [{"source": r["source"], "chunk_id": r["chunk_id"], "text": r["text"]} for r in records]
    (INDEX_DIR / "chunks_meta.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # (valgfrit) debug/inspektionsfiler
    (INDEX_DIR / "chunks.txt").write_text("\n".join(t.replace("\n", " ") for t in texts), encoding="utf-8")
    np.save(INDEX_DIR / "embeddings.npy", embs)

    # 5) Gem index-meta med korrekt signatur
    save_meta({
        "embedding_model": MODEL_NAME,
        "dim": dim,
        "num_chunks": len(chunks),
        "distance": "ip_cosine",  # oplysning til senere
    }, name="index_meta.json")

    print(f"[done] Indexed {len(chunks)} chunks with dim={dim}")
    print(f"[saved] {INDEX_DIR/'docs.index'}")
    print(f"[saved] {INDEX_DIR/'chunks_meta.json'}")
    print(f"[saved] {INDEX_DIR/'index_meta.json'}")

    # 6) Self-check
    debug_check_index(model, index, texts)

if __name__ == "__main__":
    main()
