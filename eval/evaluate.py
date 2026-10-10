import json
import statistics
import time
import requests

API_URL = "http://localhost:8000/query"
QUESTIONS_PATH = "eval/questions.json"
RESULTS_PATH = "eval/results.json"
TOP_K = 4
REFUSAL_PHRASE = "couldn't find"

def evaluate_case(case: dict) -> dict:
    start = time.perf_counter()
    response = requests.post(API_URL, json={"question": case["question"], "top_k": TOP_K}, timeout=300)
    latency_ms = round((time.perf_counter() - start) * 1000)
    response.raise_for_status()
    data = response.json()

    retrieved = [(s["pdf_name"], s["slide_num"]) for s in data["sources"]]
    expected = {(e["pdf_name"], e["slide_num"]) for e in case["expected"]}
    answer = data["answer"].replace("\u2019", "'")         
    refused = REFUSAL_PHRASE in answer.lower()

    result = {"question": case["question"], "latency_ms": latency_ms,
              "retrieved": retrieved, "answer": data["answer"]}

    if expected:   
        rank = next((i + 1 for i, slide in enumerate(retrieved) if slide in expected), None)
        result.update(type="in_scope", rank=rank, passed=rank is not None)

    else:          
        result.update(type="out_of_scope", refused=refused, passed=refused)

    return result


def main():

    with open(QUESTIONS_PATH) as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases, 1):

        r = evaluate_case(case)
        results.append(r)
        detail = f"rank={r['rank']}" if r["type"] == "in_scope" else f"refused={r['refused']}"

        print(f"[{i}/{len(cases)}] {'PASS' if r['passed'] else 'FAIL'}  {detail:<12} "
              f"{r['latency_ms']:>6} ms  {r['question']}")

    in_scope = [r for r in results if r["type"] == "in_scope"]
    out_scope = [r for r in results if r["type"] == "out_of_scope"]
    latencies = [r["latency_ms"] for r in results]

    summary = {
        "questions": len(results),
        f"hit_rate@{TOP_K}": round(sum(r["passed"] for r in in_scope) / len(in_scope), 2) if in_scope else None,
        "mrr": round(sum(1 / r["rank"] for r in in_scope if r["rank"]) / len(in_scope), 2) if in_scope else None,
        "refusal_rate": round(sum(r["refused"] for r in out_scope) / len(out_scope), 2) if out_scope else None,
        "latency_median_ms": round(statistics.median(latencies)),
        "latency_max_ms": max(latencies),
    }

    print("\n" + json.dumps(summary, indent=2))

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nFull results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()