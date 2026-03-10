# ==========================================
# backend/agents/retriever_agent.py (Fixed)
# ==========================================
import google.generativeai as genai
from pinecone import Pinecone
import os
from typing import List, Dict, Any, Optional

class RetrieverAgent:
    def __init__(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index("re-search")
        print("✅ RetrieverAgent connected to Pinecone index.")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embed_model = "models/text-embedding-004"
        print("✅ Google embedding model initialized for retrieval.")

    def embed_query(self, text: str):
        result = genai.embed_content(model=self.embed_model, content=text)
        return result["embedding"]

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top relevant chunks for the query"""
        query_vector = self.embed_query(query)  # ✅ FIXED — embed before querying

        response = self.index.query(
            namespace="research-papers",  # matches your index namespace
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for match in response.get("matches", []):
            # Pinecone returns match['id'], match['score'], match['metadata']
            meta = match.get("metadata", {}) or {}
            text = meta.get("text", "") or ""
            matches.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "metadata": meta,
                "text": text
            })

        # optional: filter by min_score
        if min_score is not None:
            matches = [m for m in matches if (m["score"] or 0) >= min_score]

        # optional: apply per-paper cap (if 'title' exists in metadata)
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
