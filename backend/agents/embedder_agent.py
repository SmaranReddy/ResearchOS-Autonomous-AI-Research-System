# ==========================================
# backend/agents/embedder_agent.py (FIXED)
# ==========================================
import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

class EmbedderAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        print("✅ Google embedding model (text-embedding-004) initialized")

    def embed_text(self, text: str):
        res = genai.embed_content(model="models/text-embedding-004", content=text)
        return res["embedding"]

    def embed_chunks(self, chunks: list[str]):
        vectors = []
        for chunk in chunks:
            res = genai.embed_content(model="models/text-embedding-004", content=chunk)
            vectors.append(res["embedding"])
        return vectors
