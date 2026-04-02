import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class Reranker:
    """
    Scores each retrieved doc for relevance to the original query using
    Groq LLM (cross-encoder style). Returns docs sorted by score descending.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def _score_doc(self, query: str, text: str) -> float:
        prompt = (
            f"Rate how relevant the following passage is to the query on a scale of 0 to 10.\n"
            f"Reply with a single number only.\n\n"
            f"Query: {query}\n\n"
            f"Passage: {text[:800]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            return float(response.choices[0].message.content.strip())
        except Exception:
            return 0.0

    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        for doc in docs:
            doc["rerank_score"] = self._score_doc(query, doc.get("text", ""))
        ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
        print(f"[Reranker] Reranked {len(ranked)} docs.")
        return ranked
