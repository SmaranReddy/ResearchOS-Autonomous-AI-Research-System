# tavily-agent.py
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import json

# ======================================
# 1️⃣ Load Environment and Initialize API
# ======================================
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("❌ Missing TAVILY_API_KEY in environment variables.")

tavily_client = TavilyClient(api_key=api_key)


# ======================================
# 2️⃣ Search Wrapper Function
# ======================================
def tavily_search(query: str, mode: str = "general", days: int = 30, max_results: int = 10):
    """
    Perform an AI-powered web search using Tavily with structured output.
    mode: 'general' | 'academic' | 'news' | 'finance'
    """

    # Domain presets for academic/news/finance research
    domain_filters = {
        "academic": [
            "arxiv.org", "nature.com", "science.org",
            "mit.edu", "stanford.edu", "researchgate.net",
        ],
        "finance": ["bloomberg.com", "ft.com", "wsj.com", "reuters.com"],
        "news": ["bbc.com", "cnn.com", "theguardian.com", "nytimes.com"],
    }

    # Map mode → topic recognized by Tavily API
    topic_map = {
        "general": "general",
        "news": "news",
        "finance": "finance",
        "academic": "general",  # academic mode uses domain filtering
    }

    # Make the API request
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        topic=topic_map.get(mode, "general"),
        days=days,
        include_answer=True,
        include_images=False,
        include_raw_content=True,
        include_domains=domain_filters.get(mode, []),
        max_results=max_results,
    )

    # Build structured result
    result = {
        "query": query,
        "mode": mode,
        "summary": response.get("answer", ""),
        "sources": [
            {"title": r["title"], "url": r["url"], "score": r.get("score")}
            for r in response.get("results", [])
        ],
    }

    # Pretty print output
    print("\n" + "=" * 80)
    print(f"🧠 Research Summary for Query: '{query}'\n")
    print(result["summary"] or "No summary available.")
    print("\n🔗 Top Sources:")
    for src in result["sources"]:
        print(f"- {src['title']} → {src['url']}")
    print("=" * 80 + "\n")

    return result


# ======================================
# 3️⃣ Example Run (Standalone)
# ======================================
if __name__ == "__main__":
    query = "Impact of AI on global labor markets"
    mode = "academic"  # can be 'academic', 'news', 'finance', 'general'
    output = tavily_search(query, mode=mode, days=30, max_results=10)

    # Save structured results to JSON
    with open("tavily_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("✅ Results saved to tavily_output.json")
