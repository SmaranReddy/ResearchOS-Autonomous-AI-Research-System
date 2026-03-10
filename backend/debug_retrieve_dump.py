# backend/debug_retrieve_dump.py
import json
from agents.retriever_agent import RetrieverAgent

def dump(query="Vision Transformers", top_k=50):
    r = RetrieverAgent()
    matches = r.retrieve(query, top_k=top_k)
    print(json.dumps(matches, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    dump()
