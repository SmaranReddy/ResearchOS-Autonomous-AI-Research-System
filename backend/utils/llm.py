# backend/agents/llm_agent.py
from groq import Groq
from dotenv import load_dotenv
import os


# ---------------------------------------------------------------------
# 🔧 Load environment variables robustly (works regardless of run path)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "../../.env")
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()  # fallback to default if .env already in cwd

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(f"❌ GROQ_API_KEY not found in .env file (checked at: {ENV_PATH})")
else:
    print("✅ GROQ_API_KEY loaded successfully")


# ---------------------------------------------------------------------
# 🧩 Initialize Groq client
# ---------------------------------------------------------------------
client = Groq(api_key=api_key)


# ---------------------------------------------------------------------
# 🧠 Base Agent Class
# ---------------------------------------------------------------------
class LLMBaseAgent:
    """Base class for Groq-powered agents using chat completions."""

    def __init__(self, model="llama-3.1-8b-instant", max_tokens=1024):
        self.model = model
        self.max_tokens = max_tokens

    def _chat(self, system_prompt, user_prompt):
        """Unified chat interface for structured prompting."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Chat completion failed: {e}")
            return "Error: Failed to generate response."


# ---------------------------------------------------------------------
# 🤖 Summarizer Agent
# ---------------------------------------------------------------------
class SummarizerAgent(LLMBaseAgent):
    """Agent that summarizes abstracts into structured sections."""

    def summarize(self, abstract: str) -> str:
        system_prompt = (
            "You are a research assistant that summarizes scientific papers "
            "into concise, structured sections."
        )
        user_prompt = (
            "Summarize the following research abstract into four sections:\n"
            "1. Problem\n2. Method\n3. Results\n4. Limitations\n\n"
            f"Abstract:\n{abstract}"
        )
        return self._chat(system_prompt, user_prompt)


# ---------------------------------------------------------------------
# 💡 Explainer Agent
# ---------------------------------------------------------------------
class ExplainerAgent(LLMBaseAgent):
    """Agent that explains technical content in simple terms."""

    def explain(self, text: str, level: str = "undergraduate") -> str:
        system_prompt = (
            "You are an expert science communicator who explains complex ideas clearly."
        )
        user_prompt = (
            f"Explain the following content to a {level}-level reader:\n\n{text}"
        )
        return self._chat(system_prompt, user_prompt)


# ---------------------------------------------------------------------
# ❓ Q&A Agent
# ---------------------------------------------------------------------
class QAAgent(LLMBaseAgent):
    """Agent that answers user questions using given context."""

    def answer(self, context: str, question: str) -> str:
        system_prompt = (
            "You are a research Q&A assistant. Use only the given context to answer accurately. "
            "If the answer is not found in the context, say 'Not found in context.'"
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
        return self._chat(system_prompt, user_prompt)
