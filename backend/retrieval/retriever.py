from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Optional

EMBED_MODEL = "all-MiniLM-L6-v2"

class RetrieverAgent:
    def __init__(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index("re-search")
        print("✅ RetrieverAgent connected to Pinecone index.")

        self.model = SentenceTransformer(EMBED_MODEL)
        print(f"✅ Local embedding model ({EMBED_MODEL}) initialized for retrieval.")

    def embed_query(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def retrieve(self, query: str, top_k: int = 5, min_score: float = None, per_paper_cap: int = None):
        """Retrieve top relevant chunks for the query"""
        query_vector = self.embed_query(query)

        response = self.index.query(
            namespace="default",
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {}) or {}
            text = meta.get("text", "") or ""
            matches.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "metadata": meta,
                "text": text
            })

        if min_score is not None:
            matches = [m for m in matches if (m["score"] or 0) >= min_score]

        if per_paper_cap is not None:
            capped = []
            counts = {}
            for m in matches:
                title = (m["metadata"] or {}).get("title", "unknown")
                if counts.get(title, 0) < per_paper_cap:
                    capped.append(m)
                    counts[title] = counts.get(title, 0) + 1
            matches = capped

        print(f"🔎 Retrieved {len(matches)} matches (requested top_k={top_k}).")
        return matches
