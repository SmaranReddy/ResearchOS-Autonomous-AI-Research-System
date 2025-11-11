# backend/agents/index_agent.py

from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

class IndexAgent:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing PINECONE_API_KEY in environment.")
        
        print("🔗 Connecting to Pinecone index...")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(
            host="re-search-02vwk3u.svc.aped-4627-b74a.pinecone.io"
        )
        print("✅ Pinecone index connection established.\n")

    def upsert_paper(self, paper, embedding):
        """Uploads a paper’s embedding and metadata (including full text) to Pinecone."""
        # Truncate full text to avoid metadata size limits (approx. 40k chars safe)
        full_text = paper.get("full_text", "")
        if len(full_text) > 40000:
            print(f"⚠️ Full text too long ({len(full_text)} chars). Truncating to 40k.")
            full_text = full_text[:40000]

        vector = {
            "id": paper["title"].replace(" ", "_")[:80],
            "values": embedding,
            "metadata": {
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "summary": paper.get("summary_structured", ""),
                "full_text": full_text,
                "link": paper.get("link", ""),
                "published": paper.get("published", ""),
                "source": "arxiv"
            }
        }

        print(f"📤 Uploading vector for: {paper['title'][:80]}")
        response = self.index.upsert(
            namespace="research-papers",
            vectors=[vector]
        )
        print(f"✅ Upsert completed for '{paper['title'][:80]}'\n")
        return response
