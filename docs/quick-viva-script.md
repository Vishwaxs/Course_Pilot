<!-- :   -->
# 🎤 Quick Viva Script — CoursePilot (3 minutes)

> Use this script for your CIA-3 viva presentation.

---

## Slide 1: Introduction (30 seconds)

> "CoursePilot is a Campus Knowledge Graph and Just-In-Time Tutor built for
> Christ University. It ingests course slides, lecture transcripts, and past
> exam papers, then lets students ask questions and get concise answers with
> full provenance — which slide, which lecture segment, and what timestamp."

**Show:** Run `streamlit run app.py` and point to the 4 tabs.

---

## Slide 2: Ingestion Pipeline (45 seconds)

> "Let me show the ingestion flow. I'll click Upload & Ingest and run the
> pipeline on our sample data."

**Demo steps:**
1. Go to **Upload & Ingest** tab.
2. Click **Run Ingestion Pipeline** (uses bundled sample data).
3. Show the progress bar and logs.
4. Point out: "It extracted 6 slides, 7 transcript segments, and 8 exam
   questions. Then it computed sentence embeddings and built a FAISS index."

---

## Slide 3: Student Chat (60 seconds)

> "Now the core feature — Student Chat."

**Demo steps:**
1. Go to **Student Chat** tab.
2. Toggle **Simulate Student Query** ON (uses canned data for instant demo).
3. Type: "What is the difference between a stack and a queue?"
4. Click **Ask**.
5. Show:
   - **Answer**: concise, 50-200 words.
   - **Provenance**: slide CS501_L1_S3 and lecture segment at 210s.
   - **Practice Questions**: matching past-paper questions.

> "Every answer comes with provenance — the exact source and timestamp.
> Students can trace back to the original material."

---

## Slide 4: Faculty Dashboard (30 seconds)

> "Faculty get a coverage dashboard."

**Demo steps:**
1. Switch role to **faculty** in the sidebar.
2. Go to **Faculty Dashboard**.
3. Show:
   - **Concept heatmap** (top-20 concepts by frequency).
   - **Gap analysis**: topics tested more than taught.
   - **Micro-lesson suggestions**.
   - **Knowledge graph** visualization.

---

## Slide 5: Limitations & Next Steps (15 seconds)

> "Current limitations: heuristic-based relations, no real SSO, local LLM
> only. Next steps would be:
> - Finer prerequisite inference with curriculum ordering.
> - Integration with campus LMS.
> - Real-time student performance tracking."

---

## Key Technical Points (if asked)

- **NLP**: spaCy noun-chunk extraction + co-occurrence heuristics.
- **Embeddings**: sentence-transformers `all-mpnet-base-v2` (768-dim).
- **Vector Store**: FAISS FlatL2 for exact k-NN.
- **Graph**: NetworkX in-memory (Neo4j optional via Docker).
- **RAG**: Template-based answer generation (local distilbart optional).
- **Privacy**: Faculty consent simulation; sample data only.

---

## Commands Cheat Sheet

```bash
# Build index
python scripts/build_faiss.py --data-dir data/sample

# Run app
streamlit run app.py

# Run tests
pytest tests/ -v

# Evaluate retrieval
python scripts/eval_retrieval.py
```
