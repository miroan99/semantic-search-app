import requests

BASE_URL = "http://127.0.0.1:8000"

def ask(question: str, top_k: int = 5):
    resp = requests.post(
        f"{BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print("Spørgsmål:", data["question"])
    print("\nSvar:\n", data["answer"])
    print("\nFørste kontekst-chunk:\n", data["context"][0][:400], "...")


if __name__ == "__main__":
    ask("Hvad sker der med ulvehunden i sneen?", top_k=5)
