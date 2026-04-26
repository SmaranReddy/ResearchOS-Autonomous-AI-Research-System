"""
Minimal baseline RAG pipeline for benchmarking.

Flow: query → embed → retrieve (single Pinecone call, top_k) → LLM → answer

Deliberately disables ALL adaptive optimizations:
- No query expansion / HyDE / multi-query
- No parallel retrieval (single Pinecone call)
- No reranking
- No retry mechanism
- No confidence scoring
- No caching
- No warmup / background ingestion
"""

import os
import time
from typing import Dict, List, Optional
from groq import Groq


class BaselineRAG:
    """Single-pass RAG with no optimizations."""

    TOP_K = 5
    MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        from retrieval.retriever import get_retriever
        self._retriever = get_retriever()
        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> Dict:
        """
        Execute the baseline pipeline and return a structured metrics dict.

        Returns
        -------
        {
            "answer":         str,
            "latency": {
                "embed_ms":    int,   # absorbed into retrieve_ms (inline)
                "retrieve_ms": int,
                "rerank_ms":   int,   # always 0 — no reranking
                "llm_ms":      int,
                "total_ms":    int,
            },
            "llm_calls":       int,   # always 1 for baseline
            "retry_triggered": bool,  # always False for baseline
            "confidence_score": None, # always None for baseline
            "docs_retrieved":  int,
            "cache_hit":       bool,  # always False for baseline
            "status":          str,
        }
        """
        t_start = time.monotonic()

        # ── Step 1: Retrieve (embed + single Pinecone query, no expansion) ──
        t_retrieve = time.monotonic()
        docs = self._retriever.retrieve(query, top_k=self.TOP_K)
        retrieve_ms = int((time.monotonic() - t_retrieve) * 1000)

        # ── Step 2: Build context from raw docs (no reranking) ──────────────
        context = self._build_context(docs)

        # ── Step 3: Single LLM call — no confidence, no retry ───────────────
        t_llm = time.monotonic()
        answer = self._generate_answer(query, context)
        llm_ms = int((time.monotonic() - t_llm) * 1000)

        total_ms = int((time.monotonic() - t_start) * 1000)

        return {
            "answer": answer,
            "latency": {
                "embed_ms": 0,
                "retrieve_ms": retrieve_ms,
                "rerank_ms": 0,
                "llm_ms": llm_ms,
                "total_ms": total_ms,
            },
            "llm_calls": 1,
            "retry_triggered": False,
            "confidence_score": None,
            "docs_retrieved": len(docs),
            "cache_hit": False,
            "status": "success" if answer and not answer.startswith("[Error") else "error",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, docs: List[Dict]) -> str:
        parts: List[str] = []
        for doc in docs:
            text = doc.get("text") or doc.get("metadata", {}).get("text", "")
            if text:
                parts.append(text.strip())
        return "\n\n---\n\n".join(parts) if parts else ""

    def _generate_answer(self, query: str, context: str) -> str:
        context_block = context if context else "No relevant documents were found."
        prompt = (
            "You are a research assistant. Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        try:
            resp = self._groq.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Error: {exc}]"
