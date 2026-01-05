# app/rag_pipeline.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv
import os

import numpy as np
try:
    import faiss
except ModuleNotFoundError:
    import faiss_cpu as faiss

MIN_SCORE = 0.15  # juster efter behov
TOP_K = 30         # standard top-k

# ⬇️ Disse utils forventes at findes fra dit indeks-arbejde (samme som du brugte i build/query)
# Sørg for at embed_text, load_faiss_index og load_meta findes i utils.py.
# Hvis dine navne afviger, så ret importerne her.
from rag_pipeline.utils import embed_text, load_faiss_index, load_meta

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or environment variables.")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_DIR = DATA / "index"

def _ensure_meta() -> dict:
    return load_meta(INDEX_DIR)

def _retrieve(
    index: faiss.Index,
    q_vec: np.ndarray,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
):
    """
    Returnerer (scores, idxs) efter:
    - evt. normalisering af query (for IndexFlatIP)
    - FAISS search
    - threshold-filtering på min_score
    """

    # Sørg for at q_vec har form (1, dim)
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    # Hvis index er IP, skal query normaliseres (cosine similarity)
    #if isinstance(index, faiss.IndexFlatIP):
    #    qn = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12
    #    q_vec = q_vec / qn
    # dobbelt normaisering

    # FAISS search
    D, I = index.search(q_vec, top_k)

    print("RAW SCORES:", D[0])
    print("IDS:", I[0])

    scores = D[0].tolist()
    idxs = I[0].tolist()

    print("[DEBUG] min_score =", min_score)
    print("[DEBUG] raw top5 =", list(zip(scores[:5], idxs[:5])))

    # Threshold-filter
    filtered = [
        (score, idx)
        for score, idx in zip(scores, idxs)
        if idx != -1 and score >= min_score
    ]

    print("[DEBUG] filtered_len =", len(filtered))
    print("[DEBUG] filtered_top =", filtered[:5])

    if not filtered:
        return [], []

    filt_scores, filt_idxs = zip(*filtered)
    return list(filt_scores), list(filt_idxs)


def _format_context(chunks, ids, scores, max_chars: int = 4000) -> str:
    # ids og scores er lister nu
    pieces = []
    total_chars = 0

    for rank, (cid, score) in enumerate(zip(ids, scores), start=1):
        meta = chunks[cid]
        text = meta["text"]

        header = (
            f"[doc#{rank} score={score:.3f} "
            f"rag_pipeline={meta['source']} chunk={meta['chunk_id']}]\n"
        )
        piece = header + text + "\n\n"

        if total_chars + len(piece) > max_chars:
            break

        pieces.append(piece)
        total_chars += len(piece)

    return "".join(pieces)

def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _print_hits(chunks, ids, scores):
    print("\n[HITS]")
    ids = _as_list(ids)
    scores = _as_list(scores)
    for rank, (cid, score) in enumerate(zip(ids, scores), start=1):
        meta = chunks[cid]
        preview = meta["text"][:120].replace("\n", " ")
        print(f"{rank}. [score={score:.3f}] {meta['source']}::chunk{meta['chunk_id']}")
        print(f"   {preview}...")

def _get_hits(chunks, ids, scores) -> list[str]:
    ids = _as_list(ids)
    scores = _as_list(scores)
    result_list = []
    for rank, (cid, score) in enumerate(zip(ids, scores), start=1):
        meta = chunks[cid]
        preview = meta["text"][:120].replace("\n", " ")
        result_list.append(f"{rank}. [score={score:.3f}] {meta['source']}::chunk{meta['chunk_id']}\n   {preview}...")
    return result_list

def _generate_answer_openai(query: str, context: str, model: str = "gpt-4o-mini") -> str:
    # Kræver: pip install openai>=1.0.0 og env var OPENAI_API_KEY sat
    from openai import OpenAI
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": (
                "Du skal besvare spørgsmålet ved at SAMMENFATTE information fra konteksten. "
                "Du må gerne drage forsigtige konklusioner, hvis de tydeligt kan udledes "
                "af flere tekststykker. Hvis noget ikke kan udledes af konteksten, så sig det."
            ),
        },
        {
            "role": "user",
            "content": (
                "Kontekst er angivet nedenfor. Hver blok starter med metadata i kantede parenteser, "
                "fx [doc#1 score=0.823 rag_pipeline=... chunk=12]. "
                "Hvis du citerer noget, så henvis til blokken ved dens doc-nummer.\n\n"
                f"{context}\n\n"
                f"Spørgsmål: {query}"
            ),
        },
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()


def run_rag(
    query: str,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    max_context_chars: int = 6000,
    debug: bool = False,
) -> Dict[str, Any]:

    # 1) Load index + chunks
    index, chunks = load_faiss_index()  # forventes at returnere (faiss.Index, List[dict])

    print("[INDEX TYPE]", type(index))
    print("[INDEX NTOTAL]", index.ntotal)

    if not isinstance(chunks, list):
        raise RuntimeError("chunks skal være en liste af dicts med mindst 'text' og 'source'.")

    # 2) Embed query
    q_vec = embed_text(query).reshape(1, -1)  # (1, dim)

    meta = _ensure_meta()

    # Example: enforce embedding dimension match early
    expected_dim = meta.get("embedding_dim")
    if expected_dim is not None and q_vec.shape[1] != expected_dim:
        raise ValueError(f"Query dim={q_vec.shape[1]} but index expects {expected_dim}. Rebuild index.")

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

    # 4a) Retrieve
    scores, ids = _retrieve(index, q_vec, top_k=top_k, min_score=MIN_SCORE)
    if not ids:
        return {"answer": "Ingen relevante dokumenter fundet (under MIN_SCORE).", "hits": []}

    # DEBUG: vis FAISS-resultater
    if debug:
        print("[FAISS IDS]", ids)
        print("[FAISS SCORES]", scores)

    # 4b) Rerank
    reranked_ids = llm_rerank(query, chunks, ids, scores, model=model)

    # 4c) Brug reranked rækkefølge
    final_ids = reranked_ids
    final_scores = [scores[ids.index(cid)] for cid in final_ids]

    #final_ids = ids
    #final_scores = scores

    print("[DEBUG] final_ids_len =", len(final_ids), "final_ids_len =", len(final_ids))

    # DEBUG: vis RERANK-resultater
    if debug:
        print("[RERANKED IDS]", final_ids)
        print("[RERANKED SCORES]", final_scores)

    # 5) Vis hits (i reranked rækkefølge)
    hit_list =_get_hits(chunks, final_ids, final_scores)

    ranked = rank_sources(final_ids, final_scores, chunks)
    print("\n[Kilder ranket]")
    for i, (src, s) in enumerate(ranked, 1):
        print(f"{i}. {src} (best={s:.3f})")

    # 6) Saml kontekst (i reranked rækkefølge)
    context = _format_context(chunks, final_ids, final_scores, max_chars=max_context_chars)

    # 7) Generér svar
    answer = _generate_answer_openai(query, context, model=model)

    # 8) Udskriv
    #print("[SVAR]")
    #print(answer)
    #print("\n[Kilder]")
    #for rank, cid in enumerate(final_ids, start=1):
    #    if cid < 0:
    #        continue
    #    ch = chunks[cid]
    #    print(f"  [{rank}] {ch.get('source','unknown')}::chunk{ch.get('chunk_id', cid)}")

    # 8) return result
    return{
        "answer": answer,
        "hits": hit_list
    }


def llm_rerank(query: str, chunks: List[Dict], ids: List[int], scores: List[float], model):
    """
    LLM-rerank på top-K hits.

    LLM'en ser en liste af dokumenter med 'idx' = position i ids-listen (0..len-1).
    Den returnerer en JSON-liste af idx'er (fx [2, 0, 1]).
    Vi mapper derefter tilbage til de rigtige FAISS-ids.
    """

    # Byg kandidatliste med lokale index (0..len(ids)-1)
    docs = []
    for local_idx, (cid, score) in enumerate(zip(ids, scores)):
        docs.append({
            "idx": local_idx,
            "score": score,
            "text": chunks[cid]["text"],
        })

    prompt = f"""
Du er en reranker. Du får et spørgsmål og en liste af dokumenter.

Hvert dokument har:
- "idx": dokumentets nummer i listen (0, 1, 2, ...)
- "score": den oprindelige FAISS-score
- "text": dokumentets tekst

Din opgave:
- Sortér dokumenterne efter relevans for spørgsmålet (bedst først).
- Svar KUN med en JSON-liste af idx-værdier, fx: [2, 0, 1]

Spørgsmål:
{query}

Dokumenter:
{json.dumps(docs, ensure_ascii=False)}

Svar KUN med en JSON-liste. Ingen forklaring, ingen tekst, kun JSON.
"""

    try:
        ranked_local_idxs = call_model_json(prompt, model)

        # Sørg for at vi kun bruger gyldige index
        ranked_local_idxs = [
            i for i in ranked_local_idxs
            if isinstance(i, int) and 0 <= i < len(ids)
        ]

        if not ranked_local_idxs:
            print("[WARN] LLM-rerank gav tom/ugyldig liste, bruger FAISS-rækkefølge.")
            return ids

        # Map tilbage til de rigtige FAISS-ids
        ranked_ids = [ids[i] for i in ranked_local_idxs]
        return ranked_ids

    except Exception as e:
        print("[WARN] LLM-rerank fejlede, bruger FAISS-rækkefølge i stedet.")
        print("       Fejl:", e)
        return ids


def call_model_json(prompt: str, model: str):
    """
    Kald LLM og forvent en JSON-liste i svaret, fx: [2, 0, 1]
    Bruges til reranking.
    """
    from openai import OpenAI
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": "Du svarer KUN med rå JSON (en liste), ingen forklaring."
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    resp = client.chat.completions.create(model=model, messages=messages)
    raw = resp.choices[0].message.content.strip()

    # Find første JSON-array i teksten
    start = raw.find("[")
    end = raw.rfind("]") + 1

    if start == -1 or end == 0:
        raise ValueError(f"LLM svarede ikke med JSON-liste: {raw!r}")

    json_str = raw[start:end]
    return json.loads(json_str)

def rank_sources_by_chunks(ids: List[int], scores: List[float], chunks: List[Dict]):
    doc_scores = defaultdict(float)

    for cid, score in zip(ids, scores):
        if cid < 0:
            continue
        src = chunks[cid]["source"]
        # brug maks-score pr. kilde
        doc_scores[src] = max(doc_scores[src], score)

    # sortér kilder efter score (højst først)
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

def rank_sources(ids: List[int], scores: List[float], chunks: List[Dict]) -> List[Tuple[str, float]]:
    src_best = defaultdict(lambda: float("-inf"))

    for cid, s in zip(ids, scores):
        if cid < 0:
            continue
        src = chunks[cid].get("source", "unknown")
        if s > src_best[src]:
            src_best[src] = s

    return sorted(src_best.items(), key=lambda x: x[1], reverse=True)

def main():
    p = argparse.ArgumentParser(description="RAG: FAISS retrieval + LLM generation")
    p.add_argument("query", nargs="?", help="Spørgsmålet du vil stille")
    p.add_argument("--topk", type=int, default=5, help="Antal chunk-hits")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chatmodel")
    p.add_argument("--repl", action="store_true", help="Interaktiv tilstand")
    p.add_argument("--debug", action="store_true", help="Vis ekstra debug-info")
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
            run_rag(q.strip(), top_k=args.topk, model=args.model, debug=args.debug)
#            print(rag_answer(q.strip()))
        return

    if not args.query:
        print("Brug: python -m rag_pipeline \"dit spørgsmål\"  (eller --repl)")
        return

    run_rag(args.query, top_k=args.topk, model=args.model, debug=args.debug)

