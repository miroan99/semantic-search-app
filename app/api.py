from fastapi import FastAPI, Query
from pydantic import BaseModel
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Local Semantic Search")

ROOT = Path(__file__).resolve().parents[1]
ST = ROOT / "storage"
META = json.loads((ST / "meta.json").read_text(encoding="utf-8"))
RECORDS = META["records"]
MODEL = SentenceTransformer(META["model"])
INDEX = faiss.read_index(str(ST / "faiss.index"))

class Hit(BaseModel):
    score: float
    id: str
    path: str
    text: str

class SearchResponse(BaseModel):
    query: str
    k: int
    hits: list[Hit]

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=2), k: int = 5):
    vec = MODEL.encode([q], normalize_embeddings=True).astype(np.float32)
    D, I = INDEX.search(vec, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        rec = RECORDS[int(idx)]
        hits.append(Hit(score=float(score), id=rec["id"], path=rec["doc_path"], text=rec["text"]))
    return SearchResponse(query=q, k=k, hits=hits)
