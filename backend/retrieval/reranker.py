import os
import re
import time
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

try:
    from core.llm_counter import record as _llm_record
except ImportError:
    def _llm_record(caller, model, elapsed_ms): pass


class Reranker:
    """
    Scores retrieved docs for relevance using a single batched LLM call.
    All documents are scored in one request instead of N sequential calls,
    reducing rerank latency from O(N * LLM_latency) to O(LLM_latency).
    """

    def __init__(self):
        # 4 s timeout — batch rerank uses max_tokens=50 and short passages;
        # should complete in <1 s normally.  Tight cap prevents blocking the pipeline.
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=4.0)
        self.model = "llama-3.1-8b-instant"

    def _batch_score(self, query: str, docs: List[Dict]) -> List[float]:
        """Score all docs in a single LLM call. Returns a list of scores (0-10)."""
        if not docs:
            return []

        # 250 chars per passage (down from 600) — halves prompt tokens while
        # preserving enough context for relevance judgement.
        passages = "\n\n".join(
            f"[{i+1}] {doc.get('text', '')[:250]}"
            for i, doc in enumerate(docs)
        )

        prompt = (
            f"Rate each passage's relevance to the query on a scale of 0 to 10.\n"
            f"Reply with ONLY a comma-separated list of numbers matching the passage order.\n"
            f"Example for 3 passages: 7,3,9\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{passages}"
        )

        _t0 = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            _elapsed = int((time.monotonic() - _t0) * 1000)
            _llm_record("Reranker._batch_score", self.model, _elapsed)
            print(f"[rerank] LLM batch score took {_elapsed}ms  ({len(docs)} docs)")
            raw = response.choices[0].message.content.strip()
            scores = [float(x.strip()) for x in re.split(r"[,\s]+", raw) if re.match(r"^\d+(\.\d+)?$", x.strip())]
            # Pad or trim to match doc count
            scores = (scores + [0.0] * len(docs))[:len(docs)]
            return scores
        except Exception as e:
            _elapsed = int((time.monotonic() - _t0) * 1000)
            print(f"[WARN] Batch rerank failed ({type(e).__name__}) — falling back to cosine similarity  ({_elapsed}ms)")
            return [min(doc.get("score", 0.0) * 10.0, 10.0) for doc in docs]

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not docs:
            return []

        # Pre-limit candidates by retrieval score to reduce LLM token cost.
        # Score at most 2*top_k docs — this avoids paying per-doc LLM cost for
        # docs we would never return anyway.
        pre_limit = min(len(docs), top_k * 2)
        candidates = (
            sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:pre_limit]
            if len(docs) > pre_limit
            else docs
        )

        scores = self._batch_score(query, candidates)
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = score

        ranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)
        top_k = min(top_k, len(ranked))
        print(f"[Reranker] Selected top {top_k} out of {len(candidates)} candidates "
              f"({len(docs)} total). Scores: {[round(d['rerank_score'], 1) for d in ranked]}")
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Module-level singleton — avoids recreating the Groq httpx connection pool
# on every pipeline call (_step_rerank used to instantiate Reranker() fresh
# each request, discarding the underlying connection pool each time).
# ---------------------------------------------------------------------------

_reranker_instance: "Reranker | None" = None


def get_reranker() -> "Reranker":
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
