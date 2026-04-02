import os
import hashlib
from collections import OrderedDict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
MAX_CACHE_SIZE = 1000
HASH_THRESHOLD = 200  # chars — use md5 key above this length


class Embedder:
    # Class-level OrderedDict — preserves insertion order for FIFO eviction
    _cache: OrderedDict = OrderedDict()

    def _make_key(self, text: str) -> str:
        normalized = text.strip().lower()
        if len(normalized) > HASH_THRESHOLD:
            return hashlib.md5(normalized.encode()).hexdigest()
        return normalized

    def _insert(self, key: str, value) -> None:
        Embedder._cache[key] = value
        if len(Embedder._cache) > MAX_CACHE_SIZE:
            Embedder._cache.popitem(last=False)  # evict oldest (FIFO)
        if len(Embedder._cache) % 100 == 0:
            print(f"[CACHE] embedding cache size: {len(Embedder._cache)}/{MAX_CACHE_SIZE}")

    def __init__(self):
        load_dotenv()
        self.model = SentenceTransformer(EMBED_MODEL)
        print(f"✅ Local embedding model ({EMBED_MODEL}, dim={EMBED_DIM}) initialized")

    def embed_text(self, text: str) -> list:
        key = self._make_key(text)
        if key in Embedder._cache:
            print(f"[CACHE HIT] embedding")
            return Embedder._cache[key]
        print(f"[CACHE MISS] embedding")
        vector = self.model.encode(text).tolist()
        self._insert(key, vector)
        return vector

    def embed_chunks(self, chunks: list[str]) -> list:
        keys = [self._make_key(c) for c in chunks]
        miss_indices = [i for i, k in enumerate(keys) if k not in Embedder._cache]

        if miss_indices:
            miss_texts = [chunks[i] for i in miss_indices]
            print(f"[CACHE] embedding — {len(chunks) - len(miss_indices)} hits, {len(miss_indices)} misses — encoding misses in batch")
            vectors = self.model.encode(miss_texts)
            for i, vector in zip(miss_indices, vectors):
                self._insert(keys[i], vector.tolist())
        else:
            print(f"[CACHE] embedding — all {len(chunks)} chunks served from cache")

        return [Embedder._cache[k] for k in keys]
