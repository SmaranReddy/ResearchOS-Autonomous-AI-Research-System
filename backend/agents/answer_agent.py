# ==========================================
# backend/agents/answer_agent.py
# ==========================================

import os
from groq import Groq
from dotenv import load_dotenv


class AnswerAgent:
    """
    Uses Groq's LLM to synthesize a coherent answer
    from the retrieved text context.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)
        print("✅ AnswerAgent ready (Groq LLM).")

    def generate_answer(self, query: str, context) -> str:
        """
        Combine retrieved chunks into a final answer using Groq.
        Accepts either List[str] or List[Dict] (ranked_docs with 'text' key).
        """
        if context and isinstance(context[0], dict):
            context = [doc.get("text", "") for doc in context]
        context_text = "\n\n".join(context)
        if len(context_text) > 15000:  # keep prompt size manageable
            context_text = context_text[:15000]

        if not context_text.strip():
            return "I don't have enough information to answer this question."

        prompt = f"""You are a research assistant. Answer the question using ONLY the context provided below.
Do NOT use any outside knowledge. Do NOT hallucinate or infer beyond what is stated.

If the context does not contain enough information to answer, respond with exactly:
"I don't have enough information to answer this question."

---
CONTEXT:
{context_text}
---

QUESTION: {query}

Respond in this structure:
**Explanation:** (2-4 sentences directly answering the question from the context)

**Key Points:**
- (bullet point from context)
- (bullet point from context)
- (add more as needed)
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a research assistant that answers only from provided context. Never fabricate information."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️ AnswerAgent failed ({e})")
            return ""

    def generate_overview(self, query, context=None):
        """Academic overview mode."""
        prompt = f"""
Write a full academic overview on the topic **{query}**, including:

1. Title  
2. Abstract  
3. Historical Evolution  
4. Architecture / Core Mechanism  
5. Applications  
6. Advantages  
7. Challenges and Limitations  
8. Future Research Directions  
9. Conclusion  
10. Keywords

Tone: formal, factual, and academic. Avoid citations.
        """
        for model in self.models:
            answer = self._call_model(prompt, model)
            if answer:
                print(f"✅ Academic overview generated using {model}")
                return answer
        return "❌ Failed to generate overview."

    def generate_precise_answer(self, question, context_summary=""):
        """Short, factual answers for user Q&A."""
        prompt = f"""
Answer precisely and concisely.

Context (if relevant):
{context_summary}

Question:
{question}

Guidelines:
- Maximum 5 sentences.
- Be direct, factual, and simple.
- Avoid academic tone or repetition.
        """
        for model in self.models:
            answer = self._call_model(prompt, model)
            if answer:
                print(f"✅ Precise Q&A generated using {model}")
                return answer
        return "❌ Could not generate response."
