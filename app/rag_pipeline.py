# app/rag_pipeline.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
import os
import openai

import numpy as np

try:
    import faiss  # GPU/standard
except ModuleNotFoundError:
    import faiss_cpu as faiss  # Windows: pip install faiss-cpu

# ⬇️ Disse utils forventes at findes fra dit indeks-arbejde (samme som du brugte i build/query)
# Sørg for at embed_text, load_faiss_index og load_meta findes i utils.py.
# Hvis dine navne afviger, så ret importerne her.
from app.utils import embed_text, load_faiss_index, load_meta

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or environment variables.")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_DIR = DATA / "index"
STORAGE = ROOT / "storage"


def _ensure_meta() -> dict:
    """Læs metadata (fx embedding_model, dim, chunk_size, overlap) hvis tilgængelig."""
    meta_path = INDEX_DIR / "index_meta.json"
    if meta_path.exists():
        import json
        return json.loads(meta_path.read_text(encoding="utf-8"))
    # Fallback hvis du kun har save_meta et andet sted
    return {}


def _retrieve(
    index: faiss.Index,
    q_vec: np.ndarray,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS søgning: returnerer (distancer, ids)"""
    # Normaliser hvis du brugte cosine (faiss IndexFlatIP + normaliserede vektorer)
    # Hvis dit index blev bygget med L2, så lad være med at normalisere her.
    # Ret gerne efter din egen opbygning.
    if isinstance(index, faiss.IndexFlatIP):
        # undgå div/0
        qn = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12
        q_vec = q_vec / qn
    D, I = index.search(q_vec, top_k)
    return D[0], I[0]


def _format_context(chunks: List[dict], ids: np.ndarray, scores: np.ndarray, max_chars: int) -> str:
    """Saml kontekst med kilde-tags."""
    items = []
    total = 0
    for rank, (cid, score) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
        if cid < 0:
            continue
        ch = chunks[cid]
        text = ch.get("text", "")
        src = f'{ch.get("source","unknown")}::chunk{ch.get("chunk_id", cid)}'
        piece = f"[{rank}] {src}\n{text}\n"
        if total + len(piece) > max_chars:
            break
        items.append(piece)
        total += len(piece)
    return "\n---\n".join(items)


def _print_hits(chunks: List[dict], ids: np.ndarray, scores: np.ndarray, preview_chars: int = 180) -> None:
    print("\n[HITS]")
    for rank, (cid, score) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
        if cid < 0:
            continue
        ch = chunks[cid]
        src = f'{ch.get("source","unknown")}::chunk{ch.get("chunk_id", cid)}'
        snippet = ch.get("text", "")[:preview_chars].replace("\n", " ")
        print(f"  {rank:>2}. {src}  score={score:.3f}")
        print(f"      {snippet}...")
    print()


def _generate_answer_openai(query: str, context: str, model: str = "gpt-4o-mini") -> str:
    # Kræver: pip install openai>=1.0.0 og env var OPENAI_API_KEY sat
    from openai import OpenAI
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": (
                "Du er en hjælper, der svarer KUN med information fra konteksten. "
                "Hvis noget ikke står i konteksten, så sig kort at det ikke fremgår."
            ),
        },
        {
            "role": "user",
            "content": f"Kontext (kilder er i klammer):\n{context}\n\nSpørgsmål: {query}",
        },
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()


def run_rag(
    query: str,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    max_context_chars: int = 6000,
) -> None:
    meta = _ensure_meta()

    # 1) Load index + chunks
    index, chunks = load_faiss_index()  # forventes at returnere (faiss.Index, List[dict])
    if not isinstance(chunks, list):
        raise RuntimeError("chunks skal være en liste af dicts med mindst 'text' og 'source'.")

    # 2) Embed query
    q_vec = embed_text(query).reshape(1, -1)  # (1, dim)

    # 3) Dimensionstjek
    idx_dim = index.d
    q_dim = q_vec.shape[1]
    if q_dim != idx_dim:
        # Hjælpende fejlbesked
        emb_model = meta.get("embedding_model", "ukendt")
        raise ValueError(
            f"Embedding-dimension mismatch: query={q_dim}, index={idx_dim}. "
            f"Rebuild index med samme embedder som i meta ('{emb_model}') og brug samme i embed_text()."
        )

    # 4) Retrieve
    scores, ids = _retrieve(index, q_vec, top_k=top_k)

    # 5) Vis hits
    _print_hits(chunks, ids, scores)

    # 6) Saml kontekst
    context = _format_context(chunks, ids, scores, max_chars=max_context_chars)

    # 7) Generér svar
    answer = _generate_answer_openai(query, context, model=model)

    # 8) Udskriv
    print("[SVAR]")
    print(answer)
    print("\n[Kilder]")
    for rank, cid in enumerate(ids.tolist(), start=1):
        if cid < 0:
            continue
        ch = chunks[cid]
        print(f"  [{rank}] {ch.get('source','unknown')}::chunk{ch.get('chunk_id', cid)}")


def call_model(context: str, query: str) -> str:
    return _generate_answer_openai(query=query, context=context, model="gpt-4o-mini")

def rag_answer(query: str) -> str:
    index, chunks = load_faiss_index()
    q_vec = embed_text(query).reshape(1, -1)

    D, I = index.search(q_vec, 5)
    ids = I[0]                      # nu: shape (5,)

    context = "\n".join(
        chunks[int(i)]["text"]      # i er nu et enkelt id
        for i in ids
        if i >= 0                   # FAISS kan returnere -1 hvis ingen hit
    )

    return call_model(context, query)

def main():
    p = argparse.ArgumentParser(description="RAG: FAISS retrieval + LLM generation")
    p.add_argument("query", nargs="?", help="Spørgsmålet du vil stille")
    p.add_argument("--topk", type=int, default=5, help="Antal chunk-hits")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chatmodel")
    p.add_argument("--repl", action="store_true", help="Interaktiv tilstand")
    args = p.parse_args()

    if args.repl and args.query:
        print("Ignorerer --repl fordi der blev givet query.")
        run_rag(args.query, top_k=args.topk, model=args.model)
        return

    if args.repl:
        print("Interaktiv RAG. Tom linje for at afslutte.")
        while True:
            try:
                q = input("\n> ")
            except EOFError:
                break
            if not q.strip():
                break
#           run_rag(q.strip(), top_k=args.topk, model=args.model)
            print(rag_answer(q.strip()))
        return

    if not args.query:
        print("Brug: python -m app.rag_pipeline \"dit spørgsmål\"  (eller --repl)")
        return

#   run_rag(args.query, top_k=args.topk, model=args.model)
    print(rag_answer(args.query))

if __name__ == "__main__":
    # Sikkerhed: tjek for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Advarsel: OPENAI_API_KEY er ikke sat. Sæt den før kørsel.")
    main()
