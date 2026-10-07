import sys
import time

from retrieval.retriever import retrieve
from generation.generator import generate


def answer(question: str, k: int = 4) -> dict:
    """Full RAG pipeline: retrieve relevant slides, then generate a cited answer."""

    t0 = time.perf_counter()
    chunks = retrieve(question, k=k)
    t1 = time.perf_counter()
    response = generate(question, chunks)
    t2 = time.perf_counter()

    return {
        "question": question,
        "answer": response,
        "sources": [
            {
                "pdf_name": c["metadata"]["pdf_name"],
                "slide_num": c["metadata"]["slide_num"],
                "distance": round(c["distance"], 3),
            }
            for c in chunks
        ],
        "timings_ms": {
            "retrieval": round((t1 - t0) * 1000),
            "generation": round((t2 - t1) * 1000),
            "total": round((t2 - t0) * 1000),
        },
    }


def main():
    # Usage: python -m src.rag "your question here"
    # With no argument, starts an interactive loop.
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = iter(lambda: input("\nAsk a question (or 'q' to quit): ").strip(), "q")

    for q in questions:
        if not q:
            continue
        result = answer(q)
        print(f"\n{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['pdf_name']} slide {s['slide_num']} (distance {s['distance']})")
        print(f"Time: {result['timings_ms']}")


if __name__ == "__main__":
    main()