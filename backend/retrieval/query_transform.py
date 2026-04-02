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

    def transform(self, query: str) -> list[str]:
        """
        Returns all query variants for multi-query retrieval:
          [original, HyDE rewrite, variation1, variation2, variation3]
        """
        hyde = self._hyde(query)
        variations = self._expand(query)
        queries = [query, hyde] + variations
        print(f"[MULTI-QUERY] generated queries: {len(queries)} (1 original + 1 HyDE + {len(variations)} variations)")
        return queries
