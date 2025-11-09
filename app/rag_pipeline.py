# app/rag_pipeline.py
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Tuple
import faiss  # standard name
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()  # loads .env automatically

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or environment variables.")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "index"

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(ST_MODEL_NAME)

print("[debug] ST dim =", st_model.get_sentence_embedding_dimension())

CHAT_MODEL = "gpt-4o-mini"

def embed_text(text: str) -> np.ndarray:
    # brug ST-model, ikke OpenAI
    emb = st_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # shape: (1, dim)
    return emb[0].astype("float32")

def load_faiss_index() -> Tuple[faiss.IndexFlatIP, list]:
    index_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "chunks.txt"

    index = faiss.read_index(str(index_path))

    # meget simpel metadata: én linje pr. chunk
    with meta_path.open("r" , encoding="utf-8") as f:
        chunks = [line.rstrip("\n") for line in f]

    return index, chunks


def retrieve_chunks(query: str, k: int = 5) -> List[str]:
    index, chunks = load_faiss_index()
    q_vec = embed_text(query).reshape(1, -1)

    if q_vec.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: query={q_vec.shape[1]}, index={index.d}. "
            "Brug samme embedding-model til at bygge index og til queries."
        )

    print("[debug] index.d =", index.d, " | q_vec.shape =", q_vec.shape)

    scores, indices = index.search(q_vec, k)
    idxs = indices[0]

    results = []
    for i in idxs:
        if 0 <= i < len(chunks):
            results.append(chunks[i])
    return results

def search_with_scores(query: str, k: int = 5):
    """Retrieve both text chunks and their FAISS similarity scores."""
    index, chunks = load_faiss_index()
    q_vec = embed_text(query).reshape(1, -1)
    scores, indices = index.search(q_vec, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if 0 <= idx < len(chunks):
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk": chunks[idx][:400].replace("\n", " ") + "..."
            })
    return results

def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "Du er en hjælpsom assistent. Brug KUN konteksten herunder til at svare. "
        "Hvis du ikke kan finde svaret i konteksten, så sig det."
    )

    user_content = (
        f"Her er konteksten:\n\n{context}\n\n"
        f"Spørgsmål: {query}\n\n"
        "Svar kort og præcist på dansk."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def answer_query(query: str, k: int = 5) -> dict:
    chunks = retrieve_chunks(query, k=k)
    answer = generate_answer(query, chunks)
    return {
        "query": query,
        "k": k,
        "context": chunks,
        "answer": answer,
    }

if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.table import Table

    console = Console()
    query = " ".join(sys.argv[1:]).strip() or "test"

    console.rule(f"[bold blue]RAG Query: {query}")

    # 🔹 Retrieve + print scores
    results = search_with_scores(query, k=5)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Chunk (preview)", width=80)

    for r in results:
        table.add_row(str(r["rank"]), f"{r['score']:.3f}", r["chunk"])
    console.print(table)

    # 🔹 Generate final answer
    answer = generate_answer(query, [r["chunk"] for r in results])
    console.rule("[bold green]Generated Answer")
    console.print(answer)
