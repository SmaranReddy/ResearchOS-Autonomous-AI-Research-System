"""
LangGraph Minimal Example (GROQ Version)
----------------------------------------
1️⃣ Normal LLM pass
2️⃣ Reflection agent critiques the output
3️⃣ LLM revises based on reflection feedback

Run:
    pip install python-dotenv groq
    python reflection_pipeline_groq.py
"""

from dataclasses import dataclass
from dotenv import load_dotenv
import os
from groq import Groq

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("⚠️ Please set GROQ_API_KEY in your .env file")

client = Groq(api_key=GROQ_API_KEY)

# -------------------------------
# Utility: call the LLM
# -------------------------------
def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> str:
    """Send a simple chat completion request to Groq"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or your preferred Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------
# Data structures
# -------------------------------
@dataclass
class LLMPassResult:
    answer: str

@dataclass
class Reflection:
    critique: str
    suggestion: str

# -------------------------------
# Step 1: Normal LLM pass
# -------------------------------
def normal_llm_pass(prompt: str) -> LLMPassResult:
    response = call_llm(f"You are a helpful assistant. Answer clearly:\n\n{prompt}")
    return LLMPassResult(answer=response)

# -------------------------------
# Step 2: Reflection agent
# -------------------------------
def reflection_agent(user_prompt: str, llm_output: str) -> Reflection:
    reflection_prompt = f"""
A user asked:
{user_prompt}

The assistant answered:
{llm_output}

Critique the answer briefly:
1. Identify what could be improved.
2. Suggest a short re-prompt or fix.
"""
    reflection_text = call_llm(reflection_prompt, temperature=0.2)
    lines = [l.strip() for l in reflection_text.splitlines() if l.strip()]
    critique = lines[0] if lines else "No critique."
    suggestion = "\n".join(lines[1:]) if len(lines) > 1 else "No suggestion."
    return Reflection(critique, suggestion)

# -------------------------------
# Step 3: Revision pass
# -------------------------------
def revision_pass(user_prompt: str, reflection: Reflection) -> str:
    revision_prompt = f"""
Improve the following answer using the feedback.

User prompt: {user_prompt}
Critique: {reflection.critique}
Suggestion: {reflection.suggestion}

Write the improved final answer:
"""
    return call_llm(revision_prompt, temperature=0.3)

# -------------------------------
# Step 4: Orchestrate pipeline
# -------------------------------
def run_pipeline(prompt: str):
    print("🔹 Step 1: Normal LLM Pass...")
    base = normal_llm_pass(prompt)

    print("\n🔹 Step 2: Reflection Agent...")
    reflection = reflection_agent(prompt, base.answer)

    print("\n🔹 Step 3: Revised Answer...")
    revised = revision_pass(prompt, reflection)

    print("\n--- RESULTS ---")
    print("\n🧠 Initial Answer:\n", base.answer)
    print("\n💬 Critique:\n", reflection.critique)
    print("\n🪞 Suggested Fix:\n", reflection.suggestion)
    print("\n✅ Revised Answer:\n", revised)

# -------------------------------
# Run example
# -------------------------------
if __name__ == "__main__":
    demo_prompt = "Explain backpropagation simply with one numeric example."
    run_pipeline(demo_prompt)
