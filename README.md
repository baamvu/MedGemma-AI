# 🩻 MedGemma Chest X-Ray Report Generation (IU X-Ray / Open-I)

Fine-tuning **`google/medgemma-4b-it`** with **QLoRA** on the open-access **IU X-Ray (Open-I)** dataset for **chest X-ray radiology report generation** (FINDINGS + IMPRESSION), then **merge adapters** to produce a **standalone offline model** usable on another machine without re-downloading the base model.

[![Dataset](https://img.shields.io/badge/Dataset-IU%20X--Ray%20(Open--I)-orange)](https://huggingface.co/datasets/ykumards/open-i)
[![Base model](https://img.shields.io/badge/Base%20model-MedGemma%204B--IT-blue)](https://huggingface.co/google/medgemma-4b-it)

---

## Demo video

- Video demo (GitHub Release): [`releases/tag/video`](https://github.com/baamvu/MedGemma-AI/releases/tag/video)

## Overview

This repo contains:

- **A Colab-friendly training notebook** to fine-tune MedGemma 4B-IT on IU X-Ray with QLoRA.
- **A merge step** to create `merged_model/` (offline inference).
- **A local Streamlit app** (`appnew.py`) to run inference with the merged model.
- **A notebook auto-fixer** (`fix_notebook.py`) to repair GitHub rendering issues caused by broken widget metadata.

### Key Features

| Feature | Description |
|---------|-------------|
| **🧩 Offline merged model** | Merge LoRA → export `merged_model/` so inference runs offline (no base model download needed) |
| **🧠 Report generation** | Generates `FINDINGS` + `IMPRESSION` from a chest X-ray image |
| **⚡ Parameter-efficient fine-tuning** | QLoRA (4-bit NF4 + LoRA) on a single GPU (T4/A100) |
| **🖥️ Local app** | Streamlit UI for inference (`streamlit run appnew.py`) |
| **🧹 GitHub notebook fix** | Auto-fix missing widget `state` key so notebook outputs render on GitHub |

---

## What’s in this repo

- **Main (MedGemma) training notebook**: `notebooks/train_colab.ipynb`
- **GitHub-friendly copy (with outputs)**: `train_colab_fixed.ipynb`
- **Notebook JSON fixer**: `fix_notebook.py`
- **Local inference app**: `appnew.py`

Optional / hackathon track:

- `notebooks/train_gemma4_unsloth.ipynb` (Gemma 4 + Unsloth experiment)

---

## Results

### Quantitative Evaluation (IU X-Ray Test Set)

| Metric | Score | Description |
|--------|-------|-------------|
| BLEU-1 | 0.358 | Unigram overlap |
| BLEU-2 | 0.239 | Bigram overlap |
| BLEU-3 | 0.163 | Trigram overlap |
| BLEU-4 | 0.109 | 4-gram overlap |
| ROUGE-L | 0.310 | Longest common subsequence |

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (8GB+ VRAM for 4-bit inference; A100/T4 for training)
- (Optional) CUDA

### Installation

```bash
git clone https://github.com/baamvu/MedGemma-AI.git
cd MedGemma-AI
pip install -r requirements.txt
```

### Download Model

After training, you will have a folder like `merged_model/` (≈ 8–10GB). Copy it to your local machine, e.g.:

```
./output/merged_model/
  config.json
  model*.safetensors
  tokenizer.json
  ...
```

Point the app to that folder (see `appnew.py`).

### Run

```bash
streamlit run appnew.py
```

Open `http://localhost:8501` → upload a chest X-ray → get an AI-generated report with clinical triage.

---

## Training

### Dataset

[IU X-Ray (Open-I)](https://huggingface.co/datasets/ykumards/open-i) — 3,851 chest X-ray studies from Indiana University Hospital with expert radiology reports.

### Fine-tuning Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `google/medgemma-4b-it` |
| Method | QLoRA (4-bit NF4 + LoRA) |
| Hardware | T4 16GB (slow) / A100 40GB (recommended) |
| Output | `final_adapter/` (LoRA) + `merged_model/` (offline model) |

### Reproduce

Open **`notebooks/train_colab.ipynb`** on Google Colab and run all cells.

If you want a GitHub-renderable notebook with outputs, use `train_colab_fixed.ipynb`.

---

## Hardware Requirements

| Component | Minimum (4-bit) | Recommended |
|-----------|-----------------|-------------|
| GPU | RTX 3050 8GB | RTX 3060 12GB+ |
| RAM | 16 GB | 32 GB |
| Storage | 15 GB | 20 GB SSD |
| OS | Windows 10/11, Linux | Ubuntu 22.04 |

---

## Project Structure

```
├── appnew.py                           # Streamlit app (local inference)
├── fix_notebook.py                     # Fix GitHub rendering for .ipynb widgets
├── notebooks/
│   ├── train_colab.ipynb               # Main: MedGemma 4B-IT QLoRA (Colab)
│   └── train_gemma4_unsloth.ipynb      # Optional: Gemma 4 + Unsloth experiment
├── train_colab_fixed.ipynb             # GitHub-friendly notebook (with outputs)
├── requirements.txt
└── README.md
```

---

## Notes (GitHub notebook outputs)

If a notebook does not render outputs on GitHub (often due to widget metadata), run:

```bash
python fix_notebook.py train_colab_fixed.ipynb
```

This repairs `metadata.widgets` by ensuring the widget state is stored under the required `state` key.

---

## Acknowledgments

- [Google / MedGemma](https://huggingface.co/google/medgemma-4b-it) — base model
- [Indiana University / NLM](https://openi.nlm.nih.gov/) — IU X-Ray dataset source
- [HuggingFace](https://huggingface.co/) — Transformers, PEFT, TRL
