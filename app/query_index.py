from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
ST = ROOT / "storage"
META = json.loads((ST / "meta.json").read_text(encoding="utf-8"))
RECORDS = META["records"]

model = SentenceTransformer(META["model"])
index = faiss.read_index(str(ST / "faiss.index"))

def search(query: str, k: int = 5):
    q = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q.astype(np.float32), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        rec = RECORDS[int(idx)]
        hits.append({
            "score": float(score),
            "id": rec["id"],
            "path": rec["doc_path"],
            "text": rec["text"]
        })
    return hits

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "password reset policy"
    for h in search(q, k=5):
        print(f"[{h['score']:.3f}] {h['id']} — {h['path']}\n{h['text']}\n")
