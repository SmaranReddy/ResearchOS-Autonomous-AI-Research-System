import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class QueryTransformer:
    """
    Rewrites the user query into a hypothetical answer passage (HyDE).
    Embedding this passage instead of the raw query improves vector recall
    for technical/research topics.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def transform(self, query: str) -> str:
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
        rewritten = response.choices[0].message.content.strip()
        print(f"[QueryTransformer] Rewritten query: {rewritten[:80]}...")
        return rewritten
