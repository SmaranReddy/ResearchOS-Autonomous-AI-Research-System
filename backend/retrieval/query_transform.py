import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class QueryTransformer:
    """
    Generates multiple query variations for multi-query retrieval.
    Returns a list: [original_query, HyDE_rewrite, variation1, variation2, variation3]
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def _hyde(self, query: str) -> str:
        """Generate a HyDE (hypothetical document) passage for the query."""
        prompt = (
            f"Write a short, dense academic passage (3-5 sentences) that directly "
            f"answers the following research question. Do not add a preamble.\n\n"
            f"Question: {query}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def _expand(self, query: str) -> list[str]:
        """Generate 3 distinct query variations."""
        prompt = (
            f"Generate exactly 3 different search queries for retrieving research papers "
            f"about the following topic. Each query should approach the topic from a different angle. "
            f"Output only the 3 queries, one per line, no numbering or extra text.\n\n"
            f"Topic: {query}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        lines = [l.strip() for l in response.choices[0].message.content.strip().splitlines() if l.strip()]
        return lines[:3]

    def _resolve_with_history(self, query: str, chat_history: list) -> str:
        """
        Rewrite a follow-up query into a self-contained question using recent history.
        Example: history=["Compare BERT and GPT"], query="how is GAN different from them"
                 → "How is GAN different from BERT and GPT?"
        Returns original query unchanged if rewriting fails.
        """
        recent = chat_history[-2:]
        history_text = "\n".join(
            f"User: {t.get('query', '')}\nAssistant: {t.get('answer', '')[:200]}"
            for t in recent
        )
        prompt = (
            f"Given the conversation history below, rewrite the follow-up question "
            f"into a fully self-contained question that can be understood without the history. "
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
                max_tokens=80,
            )
            resolved = response.choices[0].message.content.strip()
            if resolved and resolved != query:
                print(f"[QUERY_RESOLVE] '{query}' → '{resolved}'")
            return resolved or query
        except Exception:
            return query

    def transform(self, query: str, chat_history: list = None) -> list[str]:
        """
        Returns all query variants for multi-query retrieval:
          [original, HyDE rewrite, variation1, variation2, variation3]
        If chat_history is provided, the query is first resolved into a
        self-contained question so follow-ups retrieve the right documents.
        """
        resolved = (
            self._resolve_with_history(query, chat_history)
            if chat_history
            else query
        )
        hyde = self._hyde(resolved)
        variations = self._expand(resolved)
        queries = [resolved, hyde] + variations
        print(f"[MULTI-QUERY] generated queries: {len(queries)} (1 resolved + 1 HyDE + {len(variations)} variations)")
        return queries
