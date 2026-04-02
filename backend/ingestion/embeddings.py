import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

class Embedder:
    def __init__(self):
        load_dotenv()
        self.model = SentenceTransformer(EMBED_MODEL)
        print(f"✅ Local embedding model ({EMBED_MODEL}, dim={EMBED_DIM}) initialized")

    def embed_text(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def embed_chunks(self, chunks: list[str]) -> list:
        return [self.model.encode(chunk).tolist() for chunk in chunks]
