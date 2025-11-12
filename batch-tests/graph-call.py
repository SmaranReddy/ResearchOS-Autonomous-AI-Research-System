import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx

# ==========================================
# 1️⃣ Load Environment Variables
# ==========================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([GROQ_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("❌ Missing one or more environment variables")

# ==========================================
# 2️⃣ Initialize Clients
# ==========================================
groq_client = Groq(api_key=GROQ_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# ==========================================
# 3️⃣ Extract JSON More Intelligently
# ==========================================
def extract_json_from_text(text: str):
    """Extract JSON array even if wrapped inside markdown, code, or text."""
    # Remove ```python``` / ```json``` / ``` fences
    text = re.sub(r"```(?:json|python)?", "", text, flags=re.IGNORECASE).strip()

    # Look for JSON arrays explicitly inside code blocks
    matches = re.findall(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    if matches:
        for match in matches[::-1]:  # try from last to first (most likely correct)
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

    # Fallback: clean up and find any last valid JSON-looking array
    start = text.find("[[")
    end = text.rfind("]]") + 2
    if start != -1 and end > start:
        candidate = text[start:end]
        candidate = re.sub(r"\\n", "", candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []

# ==========================================
# 4️⃣ Extract Triples via Groq LLM
# ==========================================
def extract_triples(sentence: str):
    prompt = f"""
    Extract all factual (subject, relation, object) triples from this text.
    Return ONLY a valid JSON array, nothing else. Example:
    [["Albert Einstein", "discovered", "Theory of Relativity"]]

    Text: "{sentence}"
    """

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_output = completion.choices[0].message.content.strip()
    triples = extract_json_from_text(raw_output)

    if triples:
        return triples
    else:
        print("⚠️ Could not parse valid JSON from model output.\nRaw output:\n", raw_output)
        return []

# ==========================================
# 5️⃣ Neo4j Functions
# ==========================================
def insert_triples_to_neo4j(triples):
    query = """
    MERGE (a:Entity {name: $subject})
    MERGE (b:Entity {name: $object})
    MERGE (a)-[:RELATION {type: $relation}]->(b)
    """
    with neo4j_driver.session() as session:
        for triple in triples:
            if isinstance(triple, list) and len(triple) >= 3:
                sub, rel, obj = triple[:3]
                session.run(query, subject=sub, relation=rel, object=obj)

def get_graph():
    query = """
    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
    RETURN a.name AS from, r.type AS relation, b.name AS to
    """
    with neo4j_driver.session() as session:
        results = session.run(query)
        return [record.data() for record in results]

# ==========================================
# 6️⃣ Visualize Graph
# ==========================================
def visualize_graph(graph_data, output_path="knowledge_graph.html"):
    G = nx.DiGraph()

    for rel in graph_data:
        src = rel["from"]
        dst = rel["to"]
        edge_label = rel["relation"]
        G.add_node(src, label=src)
        G.add_node(dst, label=dst)
        G.add_edge(src, dst, label=edge_label)

    net = Network(height="650px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    for e in net.edges:
        e["title"] = e["label"]

    net.write_html(output_path, open_browser=False)
    print(f"\n🌐 Graph visualization saved to: {os.path.abspath(output_path)}")
    print("👉 Open it in your browser to explore interactively.\n")

# ==========================================
# 7️⃣ Main
# ==========================================
if __name__ == "__main__":
    sentence = input("Enter a sentence to build a knowledge graph: ")

    print("\n🧠 Extracting triples using Groq LLM...")
    triples = extract_triples(sentence)

    if not triples:
        print("❌ No triples extracted.")
    else:
        print(f"✅ Extracted {len(triples)} triples:")
        for t in triples:
            print("   ", t)

        print("\n📥 Inserting triples into Neo4j...")
        insert_triples_to_neo4j(triples)

        print("✅ Data inserted. Current graph:")
        graph_data = get_graph()

        for rel in graph_data:
            print(f"  ({rel['from']}) -[{rel['relation']}]-> ({rel['to']})")

        visualize_graph(graph_data)
        print("🎯 Done! Run this in Neo4j Browser:")
        print("   MATCH (n)-[r]->(m) RETURN n,r,m;")
