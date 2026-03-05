#!/bin/bash
# :   — Neo4j CSV Import Script
# What to change:
#   1. Set NEO4J_PASSWORD to match docker-compose.yml.
#   2. Ensure Neo4j is running: docker-compose up -d
#
# Usage:
#   chmod +x scripts/import_to_neo4j.sh
#   ./scripts/import_to_neo4j.sh
#
# TODO[USER_ACTION]: REPLACE_NEO4J_PASSWORD below.

set -euo pipefail

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-changeme}"  # TODO[USER_ACTION]: REPLACE_NEO4J_PASSWORD

echo "=== Neo4j CSV Import ==="
echo "URI: $NEO4J_URI"
echo "User: $NEO4J_USER"

# Create constraint
cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" \
  "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;"

echo "Importing concepts..."
cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" \
  "LOAD CSV WITH HEADERS FROM 'file:///concepts.csv' AS row
   MERGE (c:Concept {id: row.concept_id})
   SET c.label = row.label, c.frequency = toInteger(row.frequency);"

echo "Importing edges..."
cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" \
  "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
   MATCH (a:Concept {id: row.source}), (b:Concept {id: row.target})
   MERGE (a)-[r:RELATED_TO]->(b)
   SET r.relation = row.relation, r.weight = toInteger(row.weight);"

echo "✅ Import complete!"
echo "Browse: http://localhost:7474"
