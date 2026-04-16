import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class QueryTransformer:
    """
    Generates multiple query variations for multi-query retrieval.
    Returns a list: [original_query, HyDE_rewrite, variation1, variation2, variation3]
    """

    # Hard cap on each parallel LLM call — if Groq is slow, don't let
    # query_transform block the whole pipeline.
    _TRANSFORM_FUTURE_TIMEOUT = 4.0   # seconds per future

    def __init__(self):
        # 4 s HTTP timeout — HyDE (80 tokens) and expand (50 tokens) are tiny;
        # they should complete in <1 s normally, never need 10 s.
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=4.0)
        self.model = "llama-3.1-8b-instant"

    def _hyde(self, query: str) -> str:
        """Generate a short HyDE passage for semantic retrieval augmentation."""
        prompt = (
            f"Write a 2-sentence academic passage that directly answers: {query}\n"
            f"No preamble. Be concise and factual."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,   # reduced from 200 — shorter passage, faster generation
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] HyDE generation failed: {type(e).__name__} — skipping HyDE")
            return ""   # empty → caller drops it; don't fall back to query (avoids duplicate)

    def _expand(self, query: str) -> list[str]:
        """Generate 2 distinct query variations."""
        prompt = (
            f"Generate exactly 2 short search queries for finding research papers about: {query}\n"
            f"Output only the 2 queries, one per line, no numbering."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=60,   # reduced from 100 — two short queries fit in 60 tokens
            )
            lines = [l.strip() for l in response.choices[0].message.content.strip().splitlines() if l.strip()]
            return lines[:2]
        except Exception as e:
            print(f"[ERROR] Query expansion failed: {type(e).__name__} — returning empty variations")
            return []

    def _resolve_with_history(self, query: str, chat_history: list) -> str:
        """
        Rewrite a follow-up query into a self-contained question using recent history.
        Example: history=["Compare BERT and GPT"], query="how is GAN different from them"
                 → "How is GAN different from BERT and GPT?"
        Uses last 3 turns (context window limit). Returns original query on failure.
        """
        # Use last 3 turns to keep context focused and avoid noise
        recent = chat_history[-3:]
        history_text = "\n".join(
            f"User: {t.get('query', '')}\nAssistant: {t.get('answer', '')[:300]}"
            for t in recent
        )
        prompt = (
            f"Given the conversation history below, rewrite the follow-up question "
            f"into a fully self-contained question that resolves all pronouns and "
            f"references (e.g. 'it', 'them', 'that model', 'the first one') so the "
            f"question is clear without the history. "
            f"If the question is already self-contained, return it unchanged. "
            f"Return ONLY the rewritten question, nothing else.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            resolved = response.choices[0].message.content.strip()
            if not resolved:
                return query
            print(f"[QUERY_REWRITE]")
            print(f"[QUERY_REWRITE] original='{query}'")
            print(f"[QUERY_REWRITE] rewritten='{resolved}'")
            return resolved
        except Exception as e:
            print(f"[QUERY_REWRITE] rewrite failed ({e}) — using original query")
            return query

    def transform(self, query: str, chat_history: list = None) -> list[str]:
        """
        Returns all query variants for multi-query retrieval:
          [resolved, HyDE rewrite, variation1, variation2]
        If chat_history is provided, the query is first resolved into a
        standalone self-contained question (pronouns expanded) before expansion.
        index 0 is always the resolved (or original if no history) query.
        HyDE and expand run in parallel to reduce wall-clock latency.
        """
        resolved = (
            self._resolve_with_history(query, chat_history)
            if chat_history
            else query
        )
        # Single expand call — generates 2 query variations in one API call.
        # HyDE (hypothetical document) was a second parallel call that roughly
        # doubled the Groq RPM consumption; removing it stays within the 30 RPM
        # free-tier limit while preserving multi-query retrieval via expand.
        # Result: [resolved, var1, var2] — still 3 Pinecone queries.
        try:
            variations = self._expand(resolved)
        except (FutureTimeoutError, Exception) as e:
            print(f"[WARN] expand failed ({type(e).__name__}) — falling back to single query")
            variations = []

        queries = ([resolved] + variations)[:3]
        print(f"[MULTI-QUERY] {len(queries)} queries  (resolved + {len(variations)} variations, no HyDE)")
        return queries


# ---------------------------------------------------------------------------
# Module-level singleton — avoids recreating the Groq httpx connection pool
# on every pipeline call.  QueryTransformer is called on every non-cached
# request; creating a new Groq client each time discards the TCP connection.
# ---------------------------------------------------------------------------

_query_transformer_instance: "QueryTransformer | None" = None


def get_query_transformer() -> "QueryTransformer":
    global _query_transformer_instance
    if _query_transformer_instance is None:
        _query_transformer_instance = QueryTransformer()
    return _query_transformer_instance
