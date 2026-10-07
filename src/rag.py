# rag.py
import time
from retrieval.search import search
from generation.llm import generate_answer
from config import DB_DIR, LLM_MODEL

def answer(question: str, top_k: int = 4) -> dict:
    t0 = time.perf_counter()
    hits = search(question, top_k=top_k, db_dir=DB_DIR)        # step 1: find slides
    t1 = time.perf_counter()
    text = generate_answer(question, hits, LLM_MODEL)             # step 2: write answer
    t2 = time.perf_counter()

    return {
        "answer": text,
        "sources": [h["metadata"] for h in hits],             # hits are kept, not lost
        "timings_ms": {"retrieval": round((t1 - t0) * 1000),
                       "generation": round((t2 - t1) * 1000)},
    }

if __name__ == "__main__":
    result = answer("What are the assessments in this course?")
    print(result["answer"])
    print(result["sources"])
    print(result["timings_ms"])