# ==========================================
# backend/agents/answer_agent.py
# Academic Overview + Precise Q&A
# ==========================================

import os
from groq import Groq


class AnswerAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma-7b-it"
        ]
        print("✅ AnswerAgent ready (Groq LLM with academic + interactive Q&A).")

    def _call_model(self, prompt, model_name):
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Model {model_name} failed ({e})")
            return None

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
