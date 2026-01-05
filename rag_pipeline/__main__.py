# rag_pipeline/__main__.py
import argparse
from rag_pipeline.rag_pipeline import run_rag

def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--repl", action="store_true")
    args = p.parse_args()

    if args.repl:
        while True:
            q = input("> ")
            if not q.strip():
                break
            res = run_rag(q, top_k=args.topk, debug=args.debug)
            print("\n", res["answer"])
            print("\n".join(res["hits"][:args.topk]))
        return

    if not args.query:
        print("Usage: python -m rag_pipeline \"your question\"")
        return

    res = run_rag(args.query, top_k=args.topk, debug=args.debug)
    print("\n", res["answer"])
    print("\n".join(res["hits"][:args.topk]))

if __name__ == "__main__":
    main()
