# ==========================================
# backend/agents/critique_agent.py
# ==========================================
import os
from groq import Groq
from dotenv import load_dotenv

class CritiqueAgent:
    """
    Uses Groq LLM to critique and refine the AnswerAgent output.
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"  # 🔥 Best refinement model
        print("✅ CritiqueAgent ready.")

    def critique(self, answer: str) -> str:

        prompt = f"""
You are an expert academic reviewer.

Below is an AI-generated research summary. Improve it by:
- Enhancing clarity and academic tone
- Fixing factual inconsistencies
- Removing hallucinations
- Making it concise and structured
- Preserving all factual meaning

--- BEGIN ANSWER ---
{answer}
--- END ANSWER ---

Now provide your improved version below:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )

            improved = response.choices[0].message.content.strip()
            print("\n✅ Critique completed.\n")
            return improved

        except Exception as e:
            print(f"⚠️ CritiqueAgent failed: {e}")
            return "⚠️ Critique failed due to LLM error."
