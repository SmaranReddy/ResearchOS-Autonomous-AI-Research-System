# ==========================================
# backend/agents/retriever_agent.py (FIXED)
# ==========================================
import google.generativeai as genai
from pinecone import Pinecone
import os

class RetrieverAgent:
    def __init__(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index("research-papers")
        print("✅ RetrieverAgent connected to Pinecone index.")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embed_model = "models/text-embedding-004"
        print("✅ Google embedding model initialized for retrieval.")

    def embed_query(self, text: str):
        result = genai.embed_content(model=self.embed_model, content=text)
        return result["embedding"]

    def retrieve(self, query: str, top_k: int = 5):
        vector = self.embed_query(query)

        response = self.index.query(
            namespace="default",   # ❗ FIXED
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )

        chunks = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            text = meta.get("text", "")  # ❗ FIXED
            if text:
                chunks.append(text)

        print(f"🔎 Retrieved {len(chunks)} chunks.")
        return chunks
