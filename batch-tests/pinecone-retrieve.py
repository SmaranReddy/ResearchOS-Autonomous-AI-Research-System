from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
import sys

# -----------------------------
# Step 0: Load environment variables
# -----------------------------
load_dotenv()
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
index_host = "re-search-02vwk3u.svc.aped-4627-b74a.pinecone.io"  # replace with your host

try:
    index = pc.Index(host=index_host)
    print(f"✅ Connected to index at host: {index_host}")
except Exception as e:
    print("❌ Failed to connect to index:", e)
    sys.exit(1)

# -----------------------------
# Step 3: Count all records in namespace
# -----------------------------
namespace = "re-search"  # change this if needed

try:
    print(f"🔍 Listing all vector IDs from namespace '{namespace}'...")
    id_list = index.list(namespace=namespace)

    # Normalize generator/list
    if not isinstance(id_list, list):
        id_list = list(id_list)

    total_count = len(id_list)
    print(f"📊 Total vectors in namespace '{namespace}': {total_count}")

except Exception as e:
    print("❌ Failed to count vectors:", e)
    sys.exit(1)
