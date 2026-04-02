import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

from core.executor import run_pipeline

QUERIES_PATH = os.path.join(os.path.dirname(__file__), "test_queries.json")


def evaluate():
    with open(QUERIES_PATH) as f:
        test_cases = json.load(f)

    results = []
    total_keywords = 0
    total_hits = 0

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = [kw.lower() for kw in case["expected_keywords"]]

        print(f"\n[{i}/{len(test_cases)}] Query: {query}")
        answer = run_pipeline(query)
        answer_lower = answer.lower()

        hits = [kw for kw in expected if kw in answer_lower]
        score = len(hits) / len(expected) if expected else 0

        total_keywords += len(expected)
        total_hits += len(hits)

        print(f"  Keywords matched : {hits}")
        print(f"  Score            : {len(hits)}/{len(expected)} ({score:.0%})")

        results.append({"query": query, "score": score, "matched": hits, "missed": [kw for kw in expected if kw not in hits]})

    overall = total_hits / total_keywords if total_keywords else 0
    print("\n" + "=" * 60)
    print(f"OVERALL SCORE: {total_hits}/{total_keywords} keywords matched ({overall:.0%})")
    print("=" * 60)
    return results


if __name__ == "__main__":
    evaluate()
