# Out-of-distribution Awareness for Semantic Image Segmentation

---

## Overview

This repository implements a state-of-the-art segmentation pipeline for autonomous driving,
progressing through three phases:

| Phase | Steps | Topic |
|-------|-------|-------|
| **Theory** | 1 → 3 | From pixel-based semantics to mask-based panoptic with EoMT |
| **Implementation** | 4 → 5 | Baseline evaluation + parameter-efficient fine-tuning |
| **Anomaly** | 6 → 8 | OoD detection via MSP (ERFNet) and RbA (EoMT) |

---

## Repository Structure

```
vandal-project/
├── checkpoints/          # ⚠ Heavy .pth weights — in .gitignore
├── configs/
│   ├── default_eomt.yaml     # Hyperparameters for fine-tuning
│   └── anomaly_eval.yaml     # OoD evaluation config
├── data/
│   ├── cityscapes.py         # Cityscapes Dataset (19 classes)
│   ├── datasets_ood.py       # Fishyscapes & SMIYC datasets
│   └── transforms.py         # ImageNet-norm transforms
├── models/
│   ├── erfnet.py             # ERFNet (Step 1, 6, 7)
│   ├── eomt/                 # Encoder-Only Mask Transformer
│   │   └── __init__.py       # EoMT + build_eomt factory
│   └── lora.py               # LoRA injection module (Step 5)
├── scripts/
│   ├── run_finetuning.sh     # Reproducible training launcher
│   └── run_anomaly_tests.sh  # Full anomaly evaluation suite
├── utils/
│   ├── metrics.py            # mIoU, AuPRC, FPR95, AuROC
│   ├── anomaly_methods.py    # MSP, RbA, temperature scaling
│   ├── visualization.py      # Colour-coded mask visualisation
│   └── logger.py             # Console + file + W&B logging
├── train.py                  # Entry point: Step 5 fine-tuning
├── evaluate_miou.py          # Entry point: Step 4 mIoU eval
├── evaluate_anomaly.py       # Entry point: Steps 6-8 OoD eval
├── environment.yml
└── requirements.txt
```

---

## Setup

### 1. Clone & create environment

```bash
git clone <your-repo-url>
cd vandal-project

conda env create -f environment.yml
conda activate vandal
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 2. Download data

**Cityscapes** (requires free registration):
- https://www.cityscapes-dataset.com/downloads/
- Download: `leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`
- Extract to `data/cityscapes/`

**Fishyscapes Lost & Found**:
- https://fishyscapes.com/
- Extract to `data/fishyscapes/`

**SegmentMeIfYouCan (SMIYC)**:
- https://segmentmeifyoucan.com/
- Extract to `data/smiyc/`

### 3. Download pre-trained checkpoints

Place downloaded `.pth` files in `checkpoints/`:
- `erfnet_cityscapes.pth`
- `eomt_coco.pth`
- `eomt_cityscapes.pth`

---

## Usage

### Step 4 — Evaluation Baseline

```bash
# EoMT trained on Cityscapes
python evaluate_miou.py \
    --config configs/default_eomt.yaml \
    --checkpoint checkpoints/eomt_cityscapes.pth \
    --model eomt

# EoMT trained on COCO (with remapping)
python evaluate_miou.py \
    --config configs/default_eomt.yaml \
    --checkpoint checkpoints/eomt_coco.pth \
    --model eomt --is-coco
```

### Step 5 — Fine-Tuning with LoRA

```bash
# Edit configs/default_eomt.yaml first, then:
bash scripts/run_finetuning.sh

# Or directly:
python train.py --config configs/default_eomt.yaml
```

### Steps 6 & 7 — ERFNet + MSP Anomaly Detection

```bash
python evaluate_anomaly.py \
    --config configs/anomaly_eval.yaml \
    --method msp --model erfnet --dataset fishyscapes
```

### Step 8 — EoMT + RbA Anomaly Detection

```bash
# Single evaluation
python evaluate_anomaly.py \
    --config configs/anomaly_eval.yaml \
    --method rba --model eomt --dataset fishyscapes

# With temperature scaling grid search
python evaluate_anomaly.py \
    --config configs/anomaly_eval.yaml \
    --method rba --model eomt --dataset fishyscapes \
    --temperature-search
```

### Run full anomaly suite

```bash
bash scripts/run_anomaly_tests.sh
```

---

## Key Engineering Tricks (Step 5)

| Technique | Purpose | Config key |
|-----------|---------|------------|
| **Backbone freezing** | Skip updating early ViT layers | `model.freeze_backbone_layers` |
| **AMP (FP16)** | Halve memory, double speed | `train.amp: true` |
| **LoRA** | 90%+ param reduction | `lora.rank`, `lora.alpha` |
| **Gradient accumulation** | Simulate large batch | `train.gradient_accumulation_steps` |

---

## Anomaly Methods

| Method | Model | Intuition |
|--------|-------|-----------|
| **MSP** | ERFNet | `1 − max_class(softmax(logits))` — low max prob → uncertain → anomaly |
| **RbA** | EoMT | If no query claims a pixel with high confidence → rejected by all → anomaly |

**Temperature Scaling**: divide logits by scalar `T` before softmax to calibrate confidence.
Use `--temperature-search` to find the optimal `T` via fast grid search on cached logits.

---

## Results
| Model | Method | Dataset | mIoU | AuPRC |
|-------|--------|---------|------|-------|
| ERFNet| MSP    | FS      | --   | --    |
| EoMT  | RbA    | FS      | --   | --    |

---

## Required Reading

1. ERFNet — Romera et al., IEEE T-ITS 2018
2. Panoptic Segmentation — Kirillov et al., CVPR 2019
3. Mask2Former — Cheng et al., CVPR 2022
4. DINOv2 — Oquab et al., 2023
5. **EoMT** — "Your ViT is Secretly an Image Segmentation Model", CVPR 2025
6. **RbA** — "Segmenting Unknown Regions Rejected by All", ECCV 2022

---

## Citation

If you use this codebase in your work, please acknowledge the VANDAL Lab and
the original authors of EoMT, ERFNet, and RbA.
