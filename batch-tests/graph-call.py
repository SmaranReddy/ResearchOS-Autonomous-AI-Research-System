import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def extract_kg_from_text(paragraph: str):
    """
    Use Gemini to extract knowledge graph triples from unstructured text.
    """
    prompt = f"""
    You are an information extraction assistant.
    From the text below, extract all meaningful entities and relationships.
    Return output strictly as valid JSON in this format:

    {{
      "triples": [
        {{"subject": "Entity1", "relation": "relation", "object": "Entity2"}}
      ]
    }}

    Example:
    Input: "Elon Musk founded SpaceX in 2002."
    Output:
    {{
      "triples": [
        {{"subject": "Elon Musk", "relation": "founded", "object": "SpaceX"}}
      ]
    }}

    Text:
    \"\"\"{paragraph}\"\"\"
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    # Try to parse JSON safely
    try:
        # Extract JSON substring if Gemini wraps it with text
        content = response.text.strip()
        json_str = content[content.find("{"):content.rfind("}") + 1]
        data = json.loads(json_str)
        return data.get("triples", [])
    except Exception as e:
        print("⚠️ JSON parse error:", e)
        print("Gemini raw output:", response.text)
        return []

def insert_into_neo4j(triples):
    """
    Insert extracted triples into Neo4j as nodes and relationships.
    """
    with driver.session() as session:
        for t in triples:
            s, r, o = t["subject"], t["relation"], t["object"]
            session.run("""
                MERGE (a:Entity {name: $subject})
                MERGE (b:Entity {name: $object})
                MERGE (a)-[:RELATION {type: $relation}]->(b)
            """, subject=s, object=o, relation=r)

def build_knowledge_graph(paragraph: str):
    triples = extract_kg_from_text(paragraph)
    print("📘 Extracted Triples:", triples)
    if triples:
        insert_into_neo4j(triples)
        print("✅ Knowledge graph updated in Neo4j!")
    else:
        print("⚠️ No triples extracted.")

if __name__ == "__main__":
    paragraph = input("Enter a paragraph: ")
    build_knowledge_graph(paragraph)
