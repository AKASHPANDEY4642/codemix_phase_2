# 🏥 MedManglish-RAG

> **Phonetically-Aware, Citation-Forced RAG System for Medical Queries in Code-Mixed Marathi-English (Manglish).**

---

## 🌟 Overview

**MedManglish-RAG** is an end-to-end Retrieval-Augmented Generation (RAG) system specialized for code-mixed Marathi-English (Manglish) medical queries. It addresses challenges in code-mixing, phonetic transliteration variations, token tax, cross-script retrieval, and medical hallucination prevention.

### Key Capabilities:
- **Phonetic Normalization & Transliteration**: Handles Latin script Marathi phonetics (Soundex/Metaphone/Indic-transliteration).
- **Medical Entity Masking**: Preserves medical terminology in English using Aho-Corasick automaton and UMLS CUI mapping.
- **Hybrid Retrieval**: BM25 + BGE-M3 Dense embeddings fused via Reciprocal Rank Fusion (RRF) and reranked with BGE-Reranker-v2-m3.
- **Citation-Forced QLoRA Fine-Tuning**: 4-bit quantized Qwen2.5-7B fine-tuned with LoRA adapters for ALCE-style citation adherence `[1]`, `[2]`.
- **Tri-Level Safety & Risk Guardrails**: BART zero-shot classification and entropy checks for medical triage.
- **Academic Benchmark & Evaluation**: Automated evaluation pipeline (Recall@K, MRR, RAGAS, SelfCheckGPT).

---

## 📁 Repository Structure

```
medmanglish-rag/
├── api/                    # FastAPI REST API backend (endpoints: /query, /health)
├── config/                 # Centralized settings (settings.yaml, .env.example)
├── data/
│   ├── index/              # Generated FAISS and BM25 index files
│   ├── knowledge_base/     # Processed chunk corpus
│   ├── processed/          # Synthetic code-mixed QA datasets
│   ├── raw/                # Raw PubMedQA and medical corpus
│   └── splits/             # Train/Val/Test splits (JSONL)
├── logs/                   # System and SQLite execution logs
├── models/                 # Model checkpoints & QLoRA adapters
├── results/                # Evaluation benchmark CSV outputs
├── src/                    # Core RAG Modules
│   ├── data_loader.py          # Dataset parsing & loading
│   ├── evaluate.py             # 4 evaluation experiment suites
│   ├── finetune.py             # Phase 7 QLoRA training
│   ├── generate_synthetic_data.py # CMI-controlled synthetic data generator
│   ├── ingest.py               # Phase 1 Ingestion, semantic chunking & indexing
│   ├── medical_masking.py      # Aho-Corasick medical entity protection
│   ├── phonetic_normalizer.py  # Phonetic normalizer & script detector
│   ├── query_processor.py      # UnifiedQuery builder & query pipeline
│   ├── rag_pipeline.py         # End-to-end RAG inference orchestrator
│   ├── retriever.py            # Hybrid BM25 + FAISS + RRF + Reranker
│   ├── safety.py               # Zero-shot risk classifier & guardrails
│   └── tokenizer_pipeline.py   # Vocabulary expansion & token tax profiler
├── tests/                  # Pytest unit and integration test suite
├── ui/                     # Gradio web user interface
├── GPU_LAB_GUIDE.md        # 📖 Complete execution guide for GPU lab server
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

---

## ⚡ Quick Start

### 1. Setup Environment
```bash
python -m venv venv
# Linux:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Configure Environment Keys
```bash
cp config/.env.example .env
# Fill in HF_TOKEN, GROQ_API_KEY in .env
```

### 3. Build Indexes & Ingest Data
```bash
python -m src.ingest --rebuild
```

### 4. Run Interactive Web UI
```bash
python -m ui.app
```

---

## 🖥️ Running on Lab GPU Server

For detailed setup, CUDA PyTorch installation, QLoRA fine-tuning, and running the evaluation benchmarks on the lab GPU, read **[`GPU_LAB_GUIDE.md`](file:///c:/Users/kash/medmanglish-rag/GPU_LAB_GUIDE.md)**.
