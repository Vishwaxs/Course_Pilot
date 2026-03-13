# Quick Viva Script — CoursePilot (3 minutes)

> Use this script for your CIA-3 viva presentation.

---

## Slide 1: Introduction (30 seconds)

> "CoursePilot is a Document Analytics and Visualization Dashboard built with
> Streamlit. It lets users upload PDF documents and perform NLP analysis,
> generate word clouds, run image processing, and visualize concept
> relationships — all through an interactive web interface."

**Show:** Run `streamlit run app.py` and point to the 7 tabs.

---

## Slide 2: Upload & Text Processing (45 seconds)

> "Let me show the upload flow. I'll upload a sample PDF and show the
> text processing features."

**Demo steps:**
1. Go to **Upload & Process** tab.
2. Upload a PDF file.
3. Show extracted text and images.
4. Switch to **Text Processing** tab.
5. Show tokenization, POS tagging, NER, and lemmatization results.

---

## Slide 3: Visualizations (60 seconds)

> "Now the visualization features."

**Demo steps:**
1. Go to **Word Cloud & Frequency** tab.
2. Show generated word cloud and frequency bar chart.
3. Switch to **Document Analytics** tab.
4. Show TF-IDF analysis and document similarity heatmap.
5. Show sentiment analysis results.

---

## Slide 4: Image Processing & Knowledge Graph (30 seconds)

> "We also have image processing and knowledge graph visualization."

**Demo steps:**
1. Go to **Image Processing** tab.
2. Show OpenCV operations: grayscale, blur, edge detection.
3. Switch to **Knowledge Graph** tab.
4. Show the concept relationship graph built with NetworkX.

---

## Slide 5: Technical Summary (15 seconds)

> "The app uses pdfplumber for PDF extraction, spaCy and NLTK for NLP,
> OpenCV for image processing, scikit-learn for TF-IDF and similarity
> analysis, and NetworkX for graph visualization."

---

## Key Technical Points (if asked)

- **NLP**: spaCy noun-chunk extraction + NLTK tokenization, POS, NER.
- **Text Analytics**: TF-IDF with scikit-learn, cosine similarity heatmaps.
- **Sentiment**: VADER sentiment analysis via NLTK.
- **Image Processing**: OpenCV — grayscale, blur, Canny edges, thresholding, histogram equalization.
- **Graph**: NetworkX for concept knowledge graph visualization.
- **PDF Processing**: pdfplumber for text, PyMuPDF for images.

---

## Commands Cheat Sheet

```bash
# Run app
streamlit run app.py

# Run tests
pytest tests/ -v
```
