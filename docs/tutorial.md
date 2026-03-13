# Tutorial — Setting Up and Using CoursePilot

## Prerequisites

- Python 3.10+
- pip

---

## Step 1: Environment Setup

```bash
cd CIA_3

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Step 2: Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Using the App

### Upload & Process Tab
- Upload PDF documents to extract text and images.
- View extracted text and embedded images from your PDFs.

### Text Processing Tab
- Tokenization, POS tagging, Named Entity Recognition.
- Lemmatization, sentence segmentation, and n-gram analysis.

### Word Cloud & Frequency Tab
- Generate word clouds from uploaded documents.
- View word frequency distributions and top terms.

### Document Analytics Tab
- TF-IDF analysis across multiple documents.
- Document similarity heatmaps using cosine similarity.
- Sentiment analysis with VADER.

### Image Processing Tab
- Apply OpenCV operations: grayscale, blur, edge detection, thresholding.
- View color channel histograms and histogram equalization.

### Knowledge Graph Tab
- Visualize concept relationships extracted via spaCy NLP.
- Interactive NetworkX graph of document concepts.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `spacy model not found` | Run `python -m spacy download en_core_web_sm` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| App won't start | Check Python 3.10+ and all dependencies installed |
