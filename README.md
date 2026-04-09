# 🏥 Gemma 4 Chest X-Ray AI — Agentic Radiology Report Generation

**Fine-tuned Gemma 4 E4B with QLoRA for automated chest X-ray analysis, featuring clinical triage, thinking mode reasoning, and multi-study comparison — running entirely on local consumer GPUs.**

[![Gemma 4](https://img.shields.io/badge/Model-Gemma%204%20E4B-blue)](https://ai.google.dev/gemma)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-IU%20X--Ray-orange)](https://huggingface.co/datasets/ykumards/open-i)
[![Framework](https://img.shields.io/badge/Training-Unsloth-purple)](https://github.com/unslothai/unsloth)

---

## Overview

This project demonstrates that a **4-billion parameter open model** (Gemma 4 E4B), fine-tuned with parameter-efficient techniques, can generate clinically meaningful radiology reports while running **entirely offline on an 8GB GPU laptop** — making AI-assisted diagnosis accessible to clinics without cloud infrastructure.

### Key Features

| Feature | Description |
|---------|-------------|
| **🧠 Thinking Mode** | Gemma 4's reasoning capability — the model shows step-by-step clinical reasoning before generating the report |
| **🚦 Clinical Triage** | Auto-classifies severity (Normal/Abnormal/Critical), flags urgent cases, suggests follow-up |
| **🔄 Multi-Study Comparison** | Leverages Gemma 4's 128K context to compare current vs previous X-rays and detect interval changes |
| **🌐 Bilingual** | Reports in English with optional Vietnamese translation |
| **📱 Edge AI** | Runs on consumer GPUs (RTX 3050 8GB) with 4-bit quantization — no internet required |
| **⚡ Efficient Training** | QLoRA fine-tuning with Unsloth: only 0.28% parameters trained, ~1 hour on A100 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gemma 4 E4B-IT + QLoRA                   │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────────┐ │
│  │  Vision   │───▶│Projector │───▶│   Language Model       │ │
│  │ Encoder   │    │          │    │   (Gemma 4 Decoder)    │ │
│  │ (frozen)  │    └──────────┘    │   + LoRA adapters      │ │
│  └──────────┘                     └────────────────────────┘ │
│                                              │               │
│                          ┌───────────────────┼───────────┐   │
│                          ▼                   ▼           ▼   │
│                    ┌──────────┐    ┌──────────┐  ┌────────┐  │
│                    │ Thinking │    │  Report   │  │ Triage │  │
│                    │ Process  │    │ FINDINGS  │  │SEVERITY│  │
│                    │ <think>  │    │IMPRESSION │  │FOLLOWUP│  │
│                    └──────────┘    └──────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────┘
```

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

### Comparison with Prior Work on IU X-Ray

| Model | Year | BLEU-4 | ROUGE-L |
|-------|------|--------|---------|
| Jing et al. (CNN+LSTM) | 2018 | 0.090 | 0.267 |
| R2Gen | 2020 | 0.098 | 0.277 |
| R2GenCMN | 2021 | 0.112 | 0.285 |
| **Ours (Gemma 4 + QLoRA)** | **2026** | **0.109** | **0.310** |

ROUGE-L exceeds all prior specialized architectures by **+8.8%** absolute, demonstrating the advantage of pre-trained VLMs for structured text generation.

---

## Agentic Workflow

Unlike simple image-to-text models, our system implements a **multi-step clinical workflow**:

```
Input: Chest X-ray image
         │
         ▼
   ┌─────────────┐
   │  STEP 1:    │  Model reasons through each anatomical
   │  THINKING   │  structure systematically (visible to user)
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  STEP 2:    │  Structured FINDINGS + IMPRESSION
   │  REPORT     │  following radiology reporting standards
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  STEP 3:    │  SEVERITY: Normal / Abnormal / Critical
   │  TRIAGE     │  KEY_FINDINGS + FOLLOW_UP + URGENT flag
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  STEP 4:    │  Compare with previous studies if available
   │  COMPARISON │  (using 128K context for multi-image input)
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  STEP 5:    │  Optional Vietnamese translation
   │  TRANSLATE  │  for local clinical use
   └─────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (RTX 3050 or higher)
- CUDA 11.8+

### Installation

```bash
git clone https://github.com/baamvu/MedGemma-AI.git
cd MedGemma-AI
pip install -r requirements.txt
```

### Download Model

Download the fine-tuned merged model (~8 GB) and extract to `output/merged_model/`.

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
| Base model | Gemma 4 E4B-IT |
| Method | QLoRA (4-bit NF4 + LoRA r=32) |
| Trainable params | ~11.9M (0.28% of 4.3B) |
| Framework | Unsloth + TRL SFTTrainer |
| Hardware | NVIDIA A100 40GB |
| Training time | ~1 hour (3 epochs) |
| Effective batch size | 16 (4 × 4 gradient accumulation) |

### Reproduce

Open `notebooks/train_gemma4_unsloth.ipynb` on Google Colab with A100 GPU and run all cells.

---

## Hardware Requirements

| Component | Minimum (4-bit) | Recommended |
|-----------|-----------------|-------------|
| GPU | RTX 3050 8GB | RTX 3060 12GB+ |
| RAM | 16 GB | 32 GB |
| Storage | 15 GB | 20 GB SSD |
| OS | Windows 10/11, Linux | Ubuntu 22.04 |

---

## Why This Matters

> In rural clinics across developing countries, radiologist shortages mean X-rays often wait hours or days for interpretation. Our system runs on a single laptop GPU (~$800 hardware), requires no internet, and provides immediate AI-assisted analysis — enabling general practitioners to get decision support at the point of care.

**Target use case**: AI-assisted screening — automatically flag the ~5% of abnormal cases from hundreds of routine X-rays, so radiologists focus their expertise where it matters most.

---

## Project Structure

```
├── appnew.py                           # Streamlit app (agentic workflow)
├── notebooks/
│   ├── train_gemma4_unsloth.ipynb      # Training notebook (Gemma 4 + Unsloth)
│   └── train_colab.ipynb               # Legacy training notebook (MedGemma)
├── requirements.txt
└── README.md
```

---

## License

This project uses [Gemma 4](https://ai.google.dev/gemma) (Apache 2.0) and [IU X-Ray](https://huggingface.co/datasets/ykumards/open-i) (public domain).

Code is released under MIT License.

---

## Acknowledgments

- [Google DeepMind](https://deepmind.google/) — Gemma 4 model family
- [Unsloth](https://unsloth.ai/) — Efficient fine-tuning framework
- [Indiana University / NLM](https://openi.nlm.nih.gov/) — IU X-Ray dataset
- [HuggingFace](https://huggingface.co/) — Transformers, PEFT, TRL
