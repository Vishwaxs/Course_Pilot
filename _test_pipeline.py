"""Quick pipeline test to find where it fails."""
import traceback, sys
sys.path.insert(0, ".")

print("=== Step 1: PDF ingestion ===")
try:
    from backend.ingest_pdf import ingest_pdf_folder
    docs = ingest_pdf_folder("data/uploads", source_type="slide")
    print(f"  OK: {len(docs)} docs extracted")
    if docs:
        print(f"  Keys: {list(docs[0].keys())}")
        print(f"  Text preview: {docs[0]['text'][:80]!r}")
except Exception:
    traceback.print_exc()
    docs = []

if not docs:
    print("  NO DOCS — stopping.")
    sys.exit(1)

print("\n=== Step 2: Concept extraction ===")
try:
    from backend.extract_concepts import extract_concepts_from_documents, extract_relations
    concepts, concept_docs = extract_concepts_from_documents(docs)
    print(f"  OK: {len(concepts)} concepts")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 3: Relation extraction ===")
try:
    edges = extract_relations(docs, concepts, concept_docs)
    print(f"  OK: {len(edges)} edges")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 4: Save CSVs ===")
try:
    from backend.extract_concepts import save_concepts_csv, save_edges_csv
    save_concepts_csv(concepts, "data/concepts.csv")
    save_edges_csv(edges, "data/edges.csv")
    print("  OK: CSVs saved")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 5: Embeddings ===")
try:
    from backend.embeddings import embed_texts
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)
    print(f"  OK: {len(embeddings)} embeddings, dim={len(embeddings[0])}")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 6: FAISS build ===")
try:
    from backend.faiss_index import build_index
    metadata = [{"doc_id": d["doc_id"], "source_type": d["source_type"],
                 "text": d["text"], "metadata": d.get("metadata", {})} for d in docs]
    build_index(embeddings, metadata, "data/faiss_index")
    print("  OK: FAISS index built")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL STEPS PASSED ===")
