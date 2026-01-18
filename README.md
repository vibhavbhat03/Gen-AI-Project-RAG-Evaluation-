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

