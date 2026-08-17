# 🚀 MedManglish-RAG: Lab GPU Server Execution Guide

This document contains step-by-step instructions for setting up and running the **MedManglish-RAG** system on a GPU-enabled lab workstation or server (Ubuntu/Linux or Windows with NVIDIA CUDA).

---

## 📋 Table of Contents
1. [Hardware & Software Prerequisites](#1-hardware--software-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Environment Variables (.env)](#3-environment-variables-env)
4. [Execution Workflow (Phase-by-Phase)](#4-execution-workflow-phase-by-phase)
   - [Step 1: Verify Hardware & CUDA](#step-1-verify-hardware--cuda)
   - [Step 2: Run Automated Unit Tests](#step-2-run-automated-unit-tests)
   - [Step 3: Ingestion & Vector Index Building](#step-3-ingestion--vector-index-building)
   - [Step 4: QLoRA Fine-Tuning (Phase 7)](#step-4-qlora-fine-tuning-phase-7)
   - [Step 5: Full Evaluation Benchmark (Phase 8)](#step-5-full-evaluation-benchmark-phase-8)
   - [Step 6: Launch REST API Server](#step-6-launch-rest-api-server)
   - [Step 7: Launch Gradio Web UI](#step-7-launch-gradio-web-ui)
5. [Troubleshooting & GPU Tips](#5-troubleshooting--gpu-tips)

---

## 1. Hardware & Software Prerequisites

- **OS**: Ubuntu 20.04 / 22.04 LTS (Recommended) or Windows 10/11
- **GPU**: NVIDIA GPU with >= 16GB VRAM (e.g., RTX 3090, RTX 4090, A4000, A5000, A100, V100)
- **CUDA**: CUDA 11.8 or CUDA 12.1+ installed with NVIDIA Drivers
- **Python**: Python 3.10 or 3.11

---

## 2. Environment Setup

### 2.1 Create & Activate Virtual Environment

**On Linux / Ubuntu:**
```bash
cd medmanglish-rag

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel
```

**On Windows (PowerShell):**
```powershell
cd medmanglish-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
```

### 2.2 Install PyTorch with CUDA Support

Install the matching PyTorch build for your CUDA version:

**For CUDA 12.1+:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU-accelerated FAISS on Linux (CUDA 12.x):
pip install faiss-gpu-cu12
# Or for CUDA 11.x:
# pip install faiss-gpu

# (Optional) Flash Attention for faster training if supported:
# pip install flash-attn --no-build-isolation
```

---

## 3. Environment Variables (.env)

Copy the configuration template:
```bash
cp config/.env.example .env
```

Edit `.env` using your preferred editor (`nano .env` or `vim .env`):
```env
# Hugging Face Token (REQUIRED for downloading base models like Qwen / Gemma)
HF_TOKEN=hf_your_actual_token_here

# Groq API Key (REQUIRED for fast synthetic data generation & external LLM fallback)
GROQ_API_KEY=gsk_your_groq_api_key_here

# OpenAI API Key (Optional — for text-embedding-3-small fallback)
OPENAI_API_KEY=sk-your_openai_key_here

# UMLS API Key (Optional — for remote UMLS REST API expansion)
UMLS_API_KEY=your_umls_api_key_here
```

---

## 4. Execution Workflow (Phase-by-Phase)

### Step 1: Verify Hardware & CUDA
Verify that PyTorch detects your GPU:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### Step 2: Run Automated Unit Tests
Ensure all core modules, phonetics, normalizers, and masking logic pass:
```bash
pytest tests/ -v
```

---

### Step 3: Ingestion & Vector Index Building (Phase 1 & 4)
Chunk the raw datasets (`PubMedQA`, medical QA datasets), enrich with UMLS CUIs, and build both **BM25 Sparse** and **FAISS Dense** indexes with `BAAI/bge-m3`:

```bash
python -m src.ingest --rebuild
```
> **Output generated in**: `data/index/faiss_bge_m3.index`, `data/index/bm25_index.pkl`, `data/knowledge_base/`

---

### Step 4: QLoRA Fine-Tuning (Phase 7)
Fine-tune **Qwen2.5-7B-Instruct** with 4-bit quantization (BitsAndBytes) and LoRA adapters on code-mixed Manglish with ALCE-style citations `[1]`, `[2]`:

1. **Prepare formatted training dataset**:
   ```bash
   python -m src.finetune --prepare-data --max-samples 5000
   ```
2. **Launch QLoRA Training**:
   ```bash
   python -m src.finetune --train --epochs 3 --batch-size 4
   ```
> **Trained adapters saved to**: `models/qlora_adapters/`

---

### Step 5: Full Evaluation Benchmark (Phase 8)
Run the 4 thesis experiments:
- **Exp 1**: Retrieval Baselines (Recall@5, Recall@10, MRR for BM25, Dense, Hybrid RRF, Reranked)
- **Exp 2**: Architecture vs. Baselines (UnifiedQuery, Phonetic Transliteration, UMLS Expansion)
- **Exp 3**: Generation, Naturalness & Hallucination (RAGAS + SelfCheckGPT)
- **Exp 4**: Latency & Ablation Study

Run all experiments:
```bash
python -m src.evaluate --all
```
Or run individual experiments:
```bash
python -m src.evaluate --exp1
python -m src.evaluate --exp2
python -m src.evaluate --exp3
python -m src.evaluate --exp4
```
> **Results generated in**: `results/*.csv`

---

### Step 6: Launch REST API Server
Start the FastAPI production backend:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- RAG Query endpoint: `POST http://localhost:8000/query`

---

### Step 7: Launch Gradio Web UI
Launch the interactive medical QA dashboard:
```bash
python -m ui.app
```
Access UI in your browser at `http://localhost:7860`.

---

## 5. Troubleshooting & GPU Tips

| Issue | Cause | Solution |
|---|---|---|
| **CUDA Out of Memory (OOM)** | Batch size too large for VRAM | Reduce `--batch-size 2` or `--batch-size 1` in `src.finetune` and increase `gradient_accumulation_steps: 8` in `config/settings.yaml`. |
| **Gated repo error (Hugging Face)** | Model requires access permission | Accept agreement on HuggingFace for `google/gemma-2-2b` or `Qwen/Qwen2.5-7B-Instruct`, then run `huggingface-cli login` or ensure `HF_TOKEN` is set in `.env`. |
| **FAISS Segmentation Fault / DLL Error** | Incompatible FAISS binary | Ensure you installed `faiss-gpu-cu12` (Linux CUDA 12) or `faiss-cpu`. |
| **BitsAndBytes Quantization Error** | CUDA library missing | Ensure `bitsandbytes>=0.43.0` is installed and CUDA paths are in `LD_LIBRARY_PATH`. |

---

## 📌 Summary Command Cheatsheet for Lab Agent

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Ingest & Index
python -m src.ingest --rebuild

# 3. Fine-Tune QLoRA
python -m src.finetune --prepare-data
python -m src.finetune --train --epochs 3 --batch-size 4

# 4. Evaluate
python -m src.evaluate --all

# 5. Run API & UI
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
python -m ui.app
```
