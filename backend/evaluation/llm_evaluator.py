import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))


class LLMEvaluator:
    """
    Uses an LLM to score each answer on two dimensions:
      - relevance  (1-10): does the answer address the question?
      - correctness (1-10): is the content accurate and well-grounded?
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def evaluate(self, query: str, answer: str) -> dict:
        """
        Returns:
          {
            "relevance": int,       # 1-10
            "correctness": int,     # 1-10
            "relevance_reason": str,
            "correctness_reason": str
          }
        """
        prompt = f"""You are an evaluation assistant. Score the following answer to a research question.

QUESTION: {query}

ANSWER:
{answer}

Evaluate on two dimensions and respond in valid JSON only — no explanation outside the JSON:

{{
  "relevance": <int 1-10>,
  "relevance_reason": "<one sentence>",
  "correctness": <int 1-10>,
  "correctness_reason": "<one sentence>"
}}

Scoring guide:
- relevance: 10 = directly and fully answers the question, 1 = completely off-topic
- correctness: 10 = factually accurate, well-grounded, no hallucinations, 1 = mostly fabricated or wrong
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # extract JSON even if LLM wraps it in markdown
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"  [LLMEvaluator] failed: {e}")
            return {"relevance": 0, "relevance_reason": "error", "correctness": 0, "correctness_reason": "error"}
