import os
import hashlib
from collections import OrderedDict
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from utils.model import get_embedding_model, EMBED_MODEL

MAX_CACHE_SIZE = 500
HASH_THRESHOLD = 200


class RetrieverAgent:
    # Class-level OrderedDict — preserves insertion order for FIFO eviction
    _cache: OrderedDict = OrderedDict()

    def _make_key(self, query: str, top_k: int) -> str:
        normalized = query.strip().lower()
        if len(normalized) > HASH_THRESHOLD:
            normalized = hashlib.md5(normalized.encode()).hexdigest()
        return f"{normalized}::{top_k}"

    def _insert(self, key: str, value) -> None:
        RetrieverAgent._cache[key] = value
        if len(RetrieverAgent._cache) > MAX_CACHE_SIZE:
            RetrieverAgent._cache.popitem(last=False)  # evict oldest (FIFO)
        if len(RetrieverAgent._cache) % 50 == 0:
            print(f"[CACHE] retrieval cache size: {len(RetrieverAgent._cache)}/{MAX_CACHE_SIZE}")

    def __init__(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index("re-search")
        self.model = get_embedding_model()
        print("✅ RetrieverAgent connected to Pinecone index.")

    def embed_query(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def retrieve(self, query: str, top_k: int = 5, min_score: float = None, per_paper_cap: int = None):
        """Retrieve top relevant chunks for the query"""
        cache_key = self._make_key(query, top_k)
        if cache_key in RetrieverAgent._cache:
            print(f"[CACHE HIT] retrieval")
            return RetrieverAgent._cache[cache_key]
        print(f"[CACHE MISS] retrieval")

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
        self._insert(cache_key, matches)
        return matches
