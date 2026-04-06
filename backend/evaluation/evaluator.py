import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

from core.executor import run_pipeline
from evaluation.llm_evaluator import LLMEvaluator

QUERIES_PATH = os.path.join(os.path.dirname(__file__), "test_queries.json")


def evaluate():
    with open(QUERIES_PATH) as f:
        test_cases = json.load(f)

    llm_eval = LLMEvaluator()
    results = []

    total_keywords = 0
    total_hits = 0
    total_relevance = 0
    total_correctness = 0

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = [kw.lower() for kw in case["expected_keywords"]]

        print(f"\n[{i}/{len(test_cases)}] Query: {query}")
        answer, *_ = run_pipeline(query)

        # --- Baseline: keyword score ---
        answer_lower = answer.lower()
        hits = [kw for kw in expected if kw in answer_lower]
        kw_score = len(hits) / len(expected) if expected else 0
        total_keywords += len(expected)
        total_hits += len(hits)

        # --- LLM score ---
        llm_scores = llm_eval.evaluate(query, answer)
        relevance = llm_scores.get("relevance", 0)
        correctness = llm_scores.get("correctness", 0)
        total_relevance += relevance
        total_correctness += correctness

        print(f"  [Keyword]  matched : {hits}")
        print(f"  [Keyword]  score   : {len(hits)}/{len(expected)} ({kw_score:.0%})")
        print(f"  [LLM]      relevance  : {relevance}/10 — {llm_scores.get('relevance_reason', '')}")
        print(f"  [LLM]      correctness: {correctness}/10 — {llm_scores.get('correctness_reason', '')}")

        results.append({
            "query": query,
            "keyword_score": kw_score,
            "matched": hits,
            "missed": [kw for kw in expected if kw not in hits],
            "llm_relevance": relevance,
            "llm_correctness": correctness,
        })

    n = len(test_cases)
    kw_overall = total_hits / total_keywords if total_keywords else 0
    avg_relevance = total_relevance / n
    avg_correctness = total_correctness / n

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print(f"  Keyword score   : {total_hits}/{total_keywords} ({kw_overall:.0%})")
    print(f"  Avg relevance   : {avg_relevance:.1f}/10")
    print(f"  Avg correctness : {avg_correctness:.1f}/10")
    print("=" * 60)
    return results


if __name__ == "__main__":
    evaluate()
