import os
import json
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx


class KGraphAgent:
    """
    Handles:
    - Triple extraction using Groq LLM
    - Graph storage in Neo4j
    - Graph visualization (HTML)
    """

    def __init__(self):
        load_dotenv()

        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    # ---------------------------------------------------------
    # Safe JSON Extraction
    # ---------------------------------------------------------
    def _extract_json(self, text: str):
        """Extract any JSON array from raw LLM text in a robust way."""
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

        start = text.find("[")
        end = text.rfind("]")

        if start == -1 or end == -1:
            return []

        candidate = text[start:end+1]

        try:
            data = json.loads(candidate)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    # ---------------------------------------------------------
    # Triple Extraction using Groq
    # ---------------------------------------------------------
    def extract_triples(self, text: str) -> List[List[str]]:
        """Return triples like [['A', 'likes', 'B']]"""

        prompt = f"""
        Extract factual triples from this text.
        Format strictly as a JSON list:
        [
            ["Subject", "Relation", "Object"]
        ]

        Text: "{text}"
        """

        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw_output = completion.choices[0].message.content
        triples = self._extract_json(raw_output)

        return triples

    # ---------------------------------------------------------
    # Insert Triples into Neo4j
    # ---------------------------------------------------------
    def insert_triples(self, triples: List[List[str]]):
        query = """
        MERGE (a:Entity {name: $subject})
        MERGE (b:Entity {name: $object})
        MERGE (a)-[r:RELATION]->(b)
        SET r.type = $relation
        """

        with self.neo4j_driver.session() as session:
            for t in triples:
                if len(t) >= 3:
                    sub, rel, obj = t[:3]
                    session.run(query, subject=sub, relation=rel, object=obj)

    # ---------------------------------------------------------
    # Fetch full graph
    # ---------------------------------------------------------
    def get_graph(self) -> List[Dict[str, str]]:
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        RETURN a.name AS from, r.type AS relation, b.name AS to
        """

        with self.neo4j_driver.session() as session:
            results = session.run(query)
            return [record.data() for record in results]

    # ---------------------------------------------------------
    # Visualize Graph (pyvis)
    # ---------------------------------------------------------
    def visualize(self, graph_data, output_path="knowledge_graph.html"):
        G = nx.DiGraph()

        for rel in graph_data:
            src = rel["from"]
            dst = rel["to"]
            lbl = rel["relation"]
            G.add_edge(src, dst, label=lbl)

        net = Network(height="650px", width="100%", directed=True)
        net.from_nx(G)

        for e in net.edges:
            e["title"] = e.get("label", "")

        net.write_html(output_path, open_browser=False)
        return output_path
