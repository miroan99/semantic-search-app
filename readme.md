# 🧠 semantic-search-app
Local semantic search demo — using text embeddings + FAISS vector database.

---

## 🚀 Formål
Dette mini-projekt (Day 6 i *AI Application Development Plan*) viser hvordan man:
1. Genererer **tekst-embeddings** lokalt med `sentence-transformers`
2. Bygger et **FAISS vector index**
3. Søger efter semantisk lignende tekststykker
4. Tilbyder søgning via **FastAPI**

---

## ⚙️ Installation (Windows + PyCharm)
1. Opret nyt PyCharm-projekt  
   → **Project name:** `semantic-search-app`  
   → Interpreter: *New virtual environment (venv)*  

2. Åbn terminal i PyCharm:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt


## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate
   # source .venv/bin/activate
