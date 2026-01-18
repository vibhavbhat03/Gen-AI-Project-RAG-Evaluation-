# Gen-AI-Project-RAG-Evaluation-
# Chubbyemu-Style Medical Storytelling Twin (RAG + Evaluation)

## Overview
This repository contains a “digital twin” that generates Chubbyemu-inspired medical storytelling while improving factual grounding using Retrieval-Augmented Generation (RAG) and a structured evaluation workflow. The system is designed to (1) mimic a recognizable narrative voice and (2) anchor medical claims in an external medical reference corpus with citations.

---

## What This Project Demonstrates
**GenAI literacy:**
- **Prompt engineering**: structured templates for baseline vs RAG generation; multiple output modes; formatting and safety constraints.
- **RAG**: dual-corpus retrieval (style + medical facts) with FAISS and context injection.
- **Evaluation**: automated scoring for style fidelity + grounding, plus LLM-as-a-judge rubric scoring.
- **Fine-tuning (extension)**: not implemented in the current notebook, but the repo layout supports adding it later.

---

## Key Features
- **Two-source RAG**
  - **Style retrieval**: pulls narrative examples from transcript JSON to guide tone and pacing.
  - **Facts retrieval**: pulls passages from medical PDFs to ground claims and support citations.
- **Vector search with FAISS**
  - SentenceTransformer embeddings (`all-MiniLM-L6-v2`) + cosine similarity (FAISS IndexFlatIP with normalization).
- **Multiple generation modes**
  - `FULL_STORY`: long-form narrative
  - `SHORT_AWARENESS`: concise educational version
  - `MECHANISM_TAKEAWAYS`: mechanism + bullet takeaways
- **Evaluation pipeline**
  - Style metrics: marker scoring + embedding similarity + sentence-structure signals
  - Grounding metrics: citation count + “medical specifics” count + hallucination-risk proxy
  - LLM rubric scoring: structured JSON evaluation for style + factual quality

---

## Architecture (High Level)
1. **Ingest**
   - Load **style JSON** (transcript segments) from `data/style_json/`
   - Load **medical PDFs** from `data/facts_pdfs/`
2. **Preprocess**
   - Merge transcript segments into style blocks
   - Chunk PDFs into passages for retrieval
3. **Embed + Index**
   - Generate embeddings (SentenceTransformers)
   - Build **two FAISS indices**: `style` and `facts`
4. **Generate**
   - **Baseline**: prompt-only generation (no retrieval)
   - **RAG**: retrieve top-k style + top-k facts → inject into prompt → generate with citations
5. **Evaluate**
   - Compare Baseline vs RAG across modes
   - Save outputs, tables, and charts

---

## Tech Stack
- **LLM**: Google Gemini (generation + evaluation)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS
- **PDF extraction**: PyMuPDF (`fitz`) with `pdfplumber` fallback
- **Tokenization**: `tiktoken`
- **Data/metrics**: NumPy, Pandas, SciPy, scikit-learn
- **Visualization**: Matplotlib

---

## Repository Structure
The notebook generates and uses the following layout:

---

## Data Requirements

### 1) Style JSON (expected format)
Put transcript JSON files in `data/style_json/`. Each file should contain transcript segments with at least:
- `text` (and optionally `start`, `duration`)

The pipeline can construct a combined transcript if needed.

### 2) Facts PDFs
Put one or more PDFs in `data/facts_pdfs/`. These are extracted to text, chunked, embedded, and retrieved to support citation-based grounding.

---

## Setup

### Install dependencies
You can run this in Colab or locally in a virtual environment. Typical dependencies:
- `google-generativeai`
- `sentence-transformers`
- `faiss-cpu`
- `PyMuPDF`, `pdfplumber`
- `tiktoken`
- `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`

### Configure API key (Gemini)
Do **not** hardcode keys or commit them to GitHub.

Set an environment variable:
- macOS/Linux:
  - `export GOOGLE_API_KEY="YOUR_KEY"`
- Windows:
  - `setx GOOGLE_API_KEY "YOUR_KEY"`

Then read it in code:
- `GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")`

---

## How to Run
1. Add your data:
   - transcript JSON → `data/style_json/`
   - medical PDFs → `data/facts_pdfs/`
2. Open `Trial_Modified.ipynb` and run cells top-to-bottom.
3. The notebook will:
   - build chunks, embeddings, and FAISS indices
   - generate baseline + RAG outputs for multiple modes
   - run evaluation and export results to `eval/`

---

## Evaluation: What I Measure
### Style fidelity
- Narrative marker scoring (hooks, medical terminology explanations, tension/progression cues)
- Embedding similarity to a style centroid
- Sentence-structure features (short-sentence ratio, variance)

### Grounding / hallucination risk (proxy)
- Counts citations and checks citation formatting
- Counts “medical specifics” (dosages, lab values, percentages, etc.)
- Risk increases when specifics appear without supporting citations

### LLM-as-a-judge rubric scoring
A separate model returns JSON scores for:
- Style match (voice, clarity, pacing, hook)
- Factual quality (accuracy, hedging, consistency, safety)

---

## Outputs
After a full run, you’ll typically get:
- `eval/detailed_results.csv`
- `eval/comprehensive_summary.json`
- `eval/outputs/all_outputs.json`
- charts under `eval/charts/`

---

## Notes & Limitations
- This is **educational content**, not medical advice.
- Retrieval quality depends heavily on the PDFs provided.
- For production use, tighten guardrails: enforce stronger claim-to-citation alignment and refusals when evidence is missing.

---

## Future Work
- Add a re-ranker (cross-encoder) or MMR for better retrieval diversity.
- Add fine-tuning or a lightweight adapter (e.g., LoRA) for stronger style consistency.
- Expand evaluation prompts and add human rubric scoring for validation.
- Add safety filters and stricter medical claim verification.


