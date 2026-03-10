from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
import sys

# -----------------------------
# Step 0: Load environment variables
# -----------------------------
load_dotenv()

# Get Pinecone API key from .env
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    print("❌ Missing PINECONE_API_KEY in .env file")
    sys.exit(1)

# -----------------------------
# Step 1: Initialize Pinecone client
# -----------------------------
try:
    pc = Pinecone(api_key=api_key)
    print("✅ Pinecone client initialized successfully")
except Exception as e:
    print("❌ Failed to initialize Pinecone client:", e)
    sys.exit(1)

# -----------------------------
# Step 2: Connect to your index
# -----------------------------
index_host = "re-search-02vwk3u.svc.aped-4627-b74a.pinecone.io"  # Replace with actual host

try:
    index = pc.Index(host=index_host)
    print(f"✅ Connected to index at host: {index_host}")
except Exception as e:
    print("❌ Failed to connect to index:", e)
    sys.exit(1)

# -----------------------------
# Step 3: Debug - List available indexes
# -----------------------------
try:
    indexes = pc.list_indexes()
    print("ℹ️ Available indexes:", indexes)
except Exception as e:
    print("❌ Failed to list indexes:", e)

# -----------------------------
# Step 4: Helper function to generate vectors
# -----------------------------
import random

def create_vector(dim=768):
    """Generate a random 768-dim vector for Pinecone upsert."""
    return [random.uniform(-1, 1) for _ in range(dim)]

# -----------------------------
# Step 5: Create multiple chunks dynamically
# -----------------------------
num_chunks = 5  # Number of chunks to upsert
vectors_to_upsert = []

for i in range(num_chunks):
    vectors_to_upsert.append({
        "id": f"document1#chunk{i+1}",
        "values": create_vector(),  # 768-dim random vector
        "metadata": {
            "document_id": f"document{i+1}",
            "document_title": "Introduction to Vector Databases - Updated Edition",
            "chunk_number": i+1,
            "chunk_text": f"Chunk {i+1} with new content...",
            "document_url": "https://example.com/docs/document1",
            "created_at": "2024-02-15",
            "document_type": "tutorial",
            "version": "2.1"
        }
    })

# -----------------------------
# Step 6: Upsert chunks
# -----------------------------
try:
    response = index.upsert(
        namespace="re-search",
        vectors=vectors_to_upsert
    )
    print("✅ Successfully upserted document1 chunks:", response)
except Exception as e:
    print("❌ Failed to upsert vectors:", e)

# -----------------------------
# Step 7: Debug - Count vectors in namespace
# -----------------------------
try:
    stats = index.describe_index_stats(
        namespace="example-namespace"
    )
    print("ℹ️ Index stats for namespace 'example-namespace':", stats)
except Exception as e:
    print("❌ Failed to get index stats:", e)
