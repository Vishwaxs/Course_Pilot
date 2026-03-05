<!-- :   -->
# 📖 Tutorial — Setting Up and Using CoursePilot

## Prerequisites

- Python 3.10+
- pip
- (Optional) Docker for Neo4j
- (Optional) ffmpeg for Whisper audio transcription

---

## Step 1: Environment Setup

```bash
# Clone or navigate to the project
cd CIA_3

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Step 2: Configure Environment Variables

```bash
# Copy the example
cp .env.example .env
```

Edit `.env` and fill in:
- `DATA_DIR` — path to your data (default: `data/sample`)
- `NEO4J_PASSWORD` — if using Neo4j
- `OPENAI_API_KEY` — if using OpenAI for answer generation
- `GRAPH_BACKEND` — set to `networkx` (default) or `neo4j`

---

## Step 3: Build the FAISS Index

```bash
python scripts/build_faiss.py --data-dir data/sample
```

This:
1. Loads sample slides, transcripts, and past papers.
2. Computes sentence embeddings using `all-mpnet-base-v2`.
3. Saves the FAISS index to `data/faiss_index/`.

---

## Step 4: Generate Concept CSVs

```bash
python scripts/generate_neo4j_csvs.py --data-dir data/sample
```

This creates `data/concepts.csv` and `data/edges.csv` for the knowledge graph.

---

## Step 5: (Optional) Start Neo4j

```bash
# Edit docker-compose.yml to set your password
docker-compose up -d

# Browse: http://localhost:7474
# Import:
python -m backend.neo4j_import --csv-dir data/neo4j_import --backend neo4j
```

If you skip this, the app uses NetworkX (in-memory) automatically.

---

## Step 6: Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Using the App

### Upload & Ingest Tab
- Upload PDF slides, audio files, or past papers.
- Click "Run Ingestion Pipeline" to process.
- If no files uploaded, sample data is used automatically.

### Student Chat Tab
- Type a question about the course material.
- Toggle "Simulate Student Query" for instant canned results.
- View: answer, provenance (source, slide, timestamp), and practice Qs.

### Faculty Dashboard Tab
- Switch to "faculty" role in the sidebar.
- View concept coverage heatmap.
- See gap analysis: topics tested more than covered in lectures.
- Review micro-lesson suggestions.

### Admin Tab
- Switch to "admin" role.
- View pipeline logs.
- Rebuild FAISS index or import to Neo4j with one click.
- Manage faculty consent simulation.
- Check environment status (missing configs highlighted).

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Evaluating Retrieval Quality

```bash
python scripts/eval_retrieval.py
```

Outputs Recall@1, @3, @5 for queries in `data/eval/eval_queries.json`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `spacy model not found` | Run `python -m spacy download en_core_web_sm` |
| `FAISS index not found` | Run `python scripts/build_faiss.py --data-dir data/sample` |
| `Neo4j connection refused` | Run `docker-compose up -d` or set `GRAPH_BACKEND=networkx` |
| `Whisper not installed` | Run `pip install openai-whisper` (requires ffmpeg) |
| `OPENAI_API_KEY not set` | Set in `.env` or use `LLM_PROVIDER=local` |
