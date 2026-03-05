# :   — Neo4j Import & NetworkX Fallback
# What to change:
#   1. Set NEO4J_PASSWORD in .env (must match docker-compose.yml).
#   2. Run: docker-compose up -d   to start Neo4j.
#   3. Run: python -m backend.neo4j_import --csv-dir data/
#
# TODO[USER_ACTION]: set NEO4J_PASSWORD in .env before running.

"""
neo4j_import.py — Import concept/edge CSVs into Neo4j, or fall back to
an in-memory NetworkX graph for demo mode.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_MODULE_ROOT = Path(__file__).resolve().parent.parent  # CIA_3/

GRAPH_BACKEND: str = os.getenv("GRAPH_BACKEND", "networkx")
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "changeme")


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_concepts_csv(path: str = "") -> List[Dict[str, Any]]:
    """Read concepts.csv into a list of dicts."""
    if not path:
        path = str(_MODULE_ROOT / "data" / "concepts.csv")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frequency"] = int(row.get("frequency", 0))
            rows.append(row)
    return rows


def load_edges_csv(path: str = "") -> List[Dict[str, Any]]:
    """Read edges.csv into a list of dicts."""
    if not path:
        path = str(_MODULE_ROOT / "data" / "edges.csv")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["weight"] = int(row.get("weight", 1))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Neo4j import
# ---------------------------------------------------------------------------

def import_to_neo4j(
    concepts: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> None:
    """Import concepts and edges into a Neo4j database.

    Requires Neo4j to be running (see docker-compose.yml).

    TODO[USER_ACTION]: REPLACE_NEO4J_PASSWORD in .env.
    """
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        raise ImportError("neo4j driver not installed. Run: pip install neo4j")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Create uniqueness constraint
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE"
        )
        logger.info("Neo4j constraint ensured.")

        # Clear existing data
        session.run("MATCH (n:Concept) DETACH DELETE n")

        # Batch insert concepts using UNWIND
        BATCH = 500
        for i in range(0, len(concepts), BATCH):
            batch = concepts[i:i + BATCH]
            params = [{"cid": c["concept_id"], "label": c["label"], "freq": c["frequency"]} for c in batch]
            session.run(
                "UNWIND $rows AS row "
                "MERGE (n:Concept {id: row.cid}) "
                "SET n.label = row.label, n.frequency = row.freq",
                rows=params,
            )
        logger.info("Imported %d concepts into Neo4j.", len(concepts))

        # Batch insert edges using UNWIND
        for i in range(0, len(edges), BATCH):
            batch = edges[i:i + BATCH]
            params = [{"src": e["source"], "tgt": e["target"], "rel": e["relation"], "w": e["weight"]} for e in batch]
            session.run(
                "UNWIND $rows AS row "
                "MATCH (a:Concept {id: row.src}), (b:Concept {id: row.tgt}) "
                "MERGE (a)-[r:RELATED_TO]->(b) "
                "SET r.relation = row.rel, r.weight = row.w",
                rows=params,
            )
        logger.info("Imported %d edges into Neo4j.", len(edges))

    driver.close()
    logger.info("Neo4j import complete.")


# ---------------------------------------------------------------------------
# NetworkX fallback (in-memory demo mode)
# ---------------------------------------------------------------------------

_nx_graph: Optional[nx.DiGraph] = None


def build_networkx_graph(
    concepts: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> nx.DiGraph:
    """Build an in-memory NetworkX directed graph.

    Args:
        concepts: List of concept dicts.
        edges: List of edge dicts.

    Returns:
        networkx.DiGraph with concept nodes and relation edges.
    """
    global _nx_graph
    G = nx.DiGraph()
    for c in concepts:
        G.add_node(c["concept_id"], label=c["label"], frequency=c["frequency"])
    for e in edges:
        G.add_edge(
            e["source"], e["target"],
            relation=e["relation"],
            weight=e["weight"],
        )
    _nx_graph = G
    logger.info(
        "NetworkX graph: %d nodes, %d edges.",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def get_graph(
    csv_dir: str = "",
) -> nx.DiGraph:
    """Return the graph — from cache, or build from CSVs.

    Uses GRAPH_BACKEND env to decide Neo4j vs NetworkX.
    For Neo4j, also builds a local NetworkX mirror for analytics.
    """
    global _nx_graph
    if _nx_graph is not None:
        return _nx_graph

    if not csv_dir:
        csv_dir = str(_MODULE_ROOT / "data")

    concepts_path = os.path.join(csv_dir, "concepts.csv")
    edges_path = os.path.join(csv_dir, "edges.csv")

    if not Path(concepts_path).exists():
        logger.warning("concepts.csv not found at %s — returning empty graph.", concepts_path)
        _nx_graph = nx.DiGraph()
        return _nx_graph

    concepts = load_concepts_csv(concepts_path)
    edges = load_edges_csv(edges_path) if Path(edges_path).exists() else []

    if GRAPH_BACKEND == "neo4j":
        try:
            import_to_neo4j(concepts, edges)
        except Exception as exc:
            logger.warning("Neo4j import failed (%s). Using NetworkX only.", exc)

    return build_networkx_graph(concepts, edges)


def get_concept_details(concept_id: str) -> Optional[Dict[str, Any]]:
    """Return node attributes and neighbours for a concept."""
    G = get_graph()
    if concept_id not in G:
        return None
    data = dict(G.nodes[concept_id])
    data["concept_id"] = concept_id
    data["neighbours"] = [
        {"id": n, **G.edges[concept_id, n]}
        for n in G.neighbors(concept_id)
    ]
    return data


def reset_graph_cache() -> None:
    """Invalidate the cached NetworkX graph so the next get_graph() rebuilds it.

    Call this after the ingestion pipeline writes new concepts.csv / edges.csv
    so that the faculty dashboard and admin panel see fresh data.
    """
    global _nx_graph
    _nx_graph = None
    logger.info("Graph cache invalidated.")


def get_all_concepts() -> List[Dict[str, Any]]:
    """Return all concepts with their attributes."""
    G = get_graph()
    result = []
    for nid, attrs in G.nodes(data=True):
        entry = dict(attrs)
        entry["concept_id"] = nid
        entry["degree"] = G.degree(nid)
        result.append(entry)
    return sorted(result, key=lambda x: -x.get("frequency", 0))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Import CSVs into graph DB")
    parser.add_argument("--csv-dir", default="data/",
                        help="Directory with concepts.csv and edges.csv")
    parser.add_argument("--backend", default=GRAPH_BACKEND,
                        choices=["neo4j", "networkx"],
                        help="Graph backend to use")
    args = parser.parse_args()

    os.environ["GRAPH_BACKEND"] = args.backend

    G = get_graph(csv_dir=args.csv_dir)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
