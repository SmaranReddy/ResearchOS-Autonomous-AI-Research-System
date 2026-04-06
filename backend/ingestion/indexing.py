# ==========================================
# backend/agents/index_agent.py  (FIXED)
# ==========================================
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

class Indexer:
    def __init__(self, index_name="re-search"):
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing PINECONE_API_KEY in .env file")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # ❗ FIX: correct embedding dimension for Google text-embedding-004
        if self.index_name not in self.pc.list_indexes().names():
            print(f"⚙️ Creating Pinecone index: {self.index_name} ...")
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # ❗ FIXED
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            print(f"ℹ️ Pinecone index '{self.index_name}' already exists — using it.")

        self.index = self.pc.Index(self.index_name)
        print(f"✅ Pinecone index '{self.index_name}' connected.\n")

    # ------------------------------------------
    # BATCH INDEX — FIXED
    # ------------------------------------------
    def index_chunks(self, title: str, chunks: list[str], embeddings: list[list[float]], url: str = ""):
        if not chunks or not embeddings:
            print(f"⚠️ No chunks/embeddings to index for '{title}'.")
            return

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
            meta = {
                "title": title,
                "chunk_id": i,
                "text": chunk,
                "url": url,   # paper source URL stored for citation retrieval
            }
            vectors.append({
                "id": f"{title}_{i}",
                "values": emb,
                "metadata": meta,
            })

        try:
            print(f"📤 Uploading {len(vectors)} chunks for: {title}")
            self.index.upsert(vectors=vectors, namespace="default")
            print(f"✅ Indexed {len(vectors)} chunks for: {title}\n")
        except Exception as e:
            print(f"⚠️ Batch upsert failed for '{title}': {e}")
