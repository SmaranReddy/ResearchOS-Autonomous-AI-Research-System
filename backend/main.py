import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Ensure UTF-8 output on Windows (needed for emoji in print statements)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Load .env from project root before any service initializes
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path))

from core.executor import run_pipeline

if __name__ == "__main__":
    query = "Explain transformers in deep learning"
    print(f"\nQuery: {query}\n")
    answer, _, _history, _sources, _latency, _status, _trace = run_pipeline(query, num_papers=5)
    print("\n=== FINAL ANSWER ===")
    print(answer)
