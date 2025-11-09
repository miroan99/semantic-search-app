# app/rag_api.py
import os
from dotenv import load_dotenv  # optional, for .env support

# --- Load .env if present ---
load_dotenv()

# Make sure the API key is set before starting
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("[init] OPENAI_API_KEY loaded ✅")
else:
    print("[init] OPENAI_API_KEY missing ❌ — check .env or environment setup")
    raise RuntimeError("Missing OPENAI_API_KEY")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import after key is validated (so rag_pipeline can also find it)
from .rag_pipeline import answer_query
app = FastAPI(title="Day 7 – RAG QA API")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OpenAI API key. Please set the environment variable OPENAI_API_KEY."
    )

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    top_k: int
    answer: str
    context: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    try:
        result = answer_query(req.question, k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Index missing: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=result["query"],
        top_k=result["k"],
        answer=result["answer"],
        context=result["context"],
    )
