# MedGemma 4B-IT Fine-tuning for Chest X-ray Report Generation

Fine-tune [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it) with **QLoRA** on the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) dataset for automated radiology report generation.

## Requirements

- Python 3.10+
- CUDA GPU with >= 16 GB VRAM (tested on T4 / Colab / Kaggle)
- PhysioNet credentialed account (for MIMIC data)
- HuggingFace account with access to `google/medgemma-4b-it`

## Quick Start

```bash
pip install -r requirements.txt
```

## 1. Download MIMIC Data

You need **two** datasets from PhysioNet (credentialed access required):

| Dataset | URL | What it contains |
|---|---|---|
| MIMIC-CXR-JPG 2.1.0 | https://physionet.org/content/mimic-cxr-jpg/2.1.0/ | JPG images + CSV labels/splits |
| MIMIC-CXR 2.1.0 | https://physionet.org/content/mimic-cxr/2.1.0/ | Free-text radiology reports (.txt) |

### Download via wget (Linux / Colab)

```bash
# Set your PhysioNet credentials
export PHYSIONET_USER="your_username"
export PHYSIONET_PASS="your_password"

# Download MIMIC-CXR-JPG (images + CSVs)
wget -r -N -c -np \
  --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/ \
  -P data/

# Download MIMIC-CXR (reports only — just the files/ directory)
wget -r -N -c -np \
  --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
  https://physionet.org/files/mimic-cxr/2.1.0/files/ \
  -P data/
```

After downloading, your `data/` directory should look like:

```
data/
├── mimic-cxr-jpg/                     # or physionet.org/files/mimic-cxr-jpg/2.1.0/
│   ├── files/p10/p10000032/s50414267/*.jpg
│   ├── mimic-cxr-2.0.0-split.csv
│   ├── mimic-cxr-2.0.0-metadata.csv
│   └── mimic-cxr-2.0.0-chexpert.csv
├── mimic-cxr/                         # or physionet.org/files/mimic-cxr/2.1.0/
│   └── files/p10/p10000032/s50414267.txt
└── processed/                         # created by step 2
    ├── train.jsonl
    ├── validate.jsonl
    └── test.jsonl
```

> **Tip for Colab/Kaggle**: Upload the data to Google Drive and mount it, or use Kaggle datasets. Update `--mimic_cxr_jpg_dir` and `--mimic_cxr_dir` accordingly.

## 2. Prepare Data

```bash
python -m src.data_preparation \
    --mimic_cxr_jpg_dir data/mimic-cxr-jpg \
    --mimic_cxr_dir data/mimic-cxr \
    --output_dir data/processed
```

This script:
- Reads the official MIMIC-CXR split (train / validate / test)
- Filters for **frontal views only** (PA and AP)
- Pairs each image with its FINDINGS + IMPRESSION text
- Outputs `train.jsonl`, `validate.jsonl`, `test.jsonl`

## 3. Train

```bash
# Full training (may take 6-12h per epoch on T4)
python scripts/run_train.py

# Quick test with small subset
python scripts/run_train.py --max_train_samples 100 --max_val_samples 20 --epochs 1

# Custom paths (Colab example)
python scripts/run_train.py \
    --mimic_cxr_jpg_dir /content/drive/MyDrive/mimic-cxr-jpg \
    --processed_dir /content/drive/MyDrive/processed \
    --output_dir /content/drive/MyDrive/output
```

### Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--grad_accum` | 8 | Effective batch size = 1 * grad_accum |
| `--lora_r` | 16 | LoRA rank |
| `--max_length` | 512 | Max token length per sample |

The final adapter is saved to `output/final_adapter/`.

## 4. Evaluate

```bash
# Evaluate on full test set
python scripts/run_eval.py --adapter_path output/final_adapter

# Quick evaluation
python scripts/run_eval.py --adapter_path output/final_adapter --max_samples 50

# Save metrics to JSON
python scripts/run_eval.py \
    --adapter_path output/final_adapter \
    --output_json output/metrics.json
```

Metrics reported: BLEU-1/2/3/4, ROUGE-L, CheXpert keyword F1.

## 5. Inference App

```bash
streamlit run app.py
```

The app auto-detects the trained adapter at `output/final_adapter/`. If not found, it falls back to the base MedGemma model.

## Project Structure

```
├── configs/
│   └── training_config.py      # All configuration dataclasses
├── src/
│   ├── data_preparation.py     # MIMIC-CXR -> JSONL converter
│   ├── dataset.py              # PyTorch Dataset for training
│   ├── model_setup.py          # 4-bit model + QLoRA adapter setup
│   ├── train.py                # HF Trainer loop
│   └── evaluate.py             # BLEU / ROUGE-L / CheXpert F1
├── scripts/
│   ├── run_train.py            # Training entry point
│   └── run_eval.py             # Evaluation entry point
├── app.py                      # Streamlit inference app
└── requirements.txt
```

## Notes

- **T4 GPUs** do not support native BF16 — the config auto-detects this and uses FP16 instead.
- Training on the full MIMIC-CXR dataset (~150K frontal images) takes roughly 6-12 hours per epoch on a single T4.
- For faster iteration, use `--max_train_samples` to train on a subset first.
- The vision encoder (SigLIP) is **frozen** — only language model LoRA adapters are trained.
