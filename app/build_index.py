from datetime import datetime
from pathlib import Path
import numpy as np
from sympy.codegen.cnodes import sizeof
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
try:
    import faiss  # standard name
except ModuleNotFoundError:
    import faiss_cpu as faiss  # Windows fallback

from rag_pipeline.utils import iter_texts, chunk_text, save_meta, normalize_ws

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_docs"
STORAGE = ROOT / "storage"
INDEX_DIR = ROOT / "data" / "index"

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

def build_faiss_index():
    model = SentenceTransformer(MODEL_NAME)

    texts = []
    metas = []

    for path, text in iter_texts(RAW):
        for chunk_id, chunk in enumerate(chunk_text(text)):
            chunk = normalize_ws(chunk)
            texts.append(chunk)
            metas.append(
                {
                    "source": str(path),
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    # Embeddings
    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # vi styrer selv normalisering
    ).astype("float32")

    dim = embeddings.shape[1]

    # Vælg index-type
    # Brug IP + normalisering hvis du vil have cosine-lignende adfærd
    use_ip = True  # sæt til False hvis du vil bruge L2

    if use_ip:
        index = faiss.IndexFlatIP(dim)
        # *** VIGTIGT ***: normalisér embeddings til længde 1 (L2-norm)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
    else:
        index = faiss.IndexFlatL2(dim)
        # ingen normalisering for L2

    print("Adding vectors to index ...")
    index.add(embeddings)

    # Gem index + metadata
    chunk_stat = chunk_text.__defaults__  # (800, 100)

    meta = {
        "embedding_model": MODEL_NAME,
        "embedding_dim": int(index.d),
        "chunk_size": chunk_stat[0],
        "chunk_overlap": chunk_stat[1],
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "num_chunks": len(texts),
    }
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "docs.index"))
    save_meta(INDEX_DIR / "meta.json", meta)
    print("Done. Index size:", index.ntotal)

if __name__ == "__main__":
    build_faiss_index()
