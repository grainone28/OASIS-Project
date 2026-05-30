# OASIS — Comprehensive Road Scene Understanding for Autonomous Driving

Out-of-distribution (OoD) anomaly segmentation on driving scenes, built on top of
[ERFNet](https://github.com/Eromera/erfnet_pytorch) (pixel-based baseline) and
[EoMT](https://github.com/tue-mps/eomt) (mask-based, *Your ViT is Secretly an
Image Segmentation Model*, CVPR 2025).

The project follows the 8-step track defined in the course brief
(*Comprehensive Road Scene Understanding for Autonomous Driving* — VANDAL Lab,
PoliTO):

1. Semantic segmentation with ERFNet
2. Panoptic / instance segmentation theory
3. Mask architectures (MaskFormer → Mask2Former → EoMT + DINOv2)
4. Quantitative comparison of the two pre-trained EoMTs (COCO-panoptic vs.
   Cityscapes-semantic) on Cityscapes-val
5. Fine-tuning the COCO-pretrained EoMT on Cityscapes (semantic)
6. Anomaly segmentation task & post-hoc methods
7. **Pixel baselines** — ERFNet + MSP / MaxLogit / MaxEntropy
8. **Mask baselines** — EoMT + MSP / MaxLogit / MaxEntropy / RbA, plus
   temperature scaling

---

## Repository layout

```
OASIS-Project/
├── configs/
│   └── anomaly_eval.yaml          # dataset & checkpoint paths for Steps 7-8
├── data/
│   ├── cityscapes.py              # Cityscapes loader + COCO→CS label LUT
│   ├── datasets_ood.py            # Fishyscapes (L&F, Static), SMIYC, Road Anomaly
│   └── transforms.py              # val transforms (ImageNet / ERFNet norm)
├── training/                      # EoMT TRAINING stack (PyTorch Lightning,
│   │                              #   vendored from tue-mps/eomt)
│   ├── main.py                    # Lightning CLI entry point
│   ├── configs/
│   │   ├── cityscapes_semantic.yaml
│   │   ├── phase1.yaml            # frozen-backbone fine-tune from COCO
│   │   └── phase2.yaml            # progressive unfreezing
│   ├── datasets/                  # Lightning data modules
│   ├── models/
│   │   ├── erfnet.py              # ERFNet implementation
│   │   ├── eomt.py                # EoMT model
│   │   ├── vit.py                 # DINOv2 ViT backbone wrapper
│   │   └── scale_block.py
│   └── train/                     # loss, schedule, lightning_module
├── scripts/
│   └── adapt_coco_checkpoint.py   # adapt eomt_coco.bin (640², 200q, 133cls)
│                                  #   → Cityscapes shape (1024², 100q, 19cls)
├── utils/
│   ├── anomaly_methods.py         # MSP, MaxLogit, MaxEntropy, RbA, T-scaling
│   ├── metrics.py                 # mIoU, AuPRC, FPR95, AuROC
│   ├── postprocessing.py          # logits → semantic mask helpers
│   ├── visualization.py           # colour-coded prediction & anomaly maps
│   └── logger.py
├── evaluate_miou.py               # entry point — Step 4
├── evaluate_anomaly.py            # entry point — Steps 7 & 8
├── environment.yml / requirements.txt
└── LICENSE
```

Model code (ERFNet, EoMT, ViT) lives only under `training/models/` and is
imported from there by both the Lightning training pipeline and the
`evaluate_*.py` scripts — no duplication.

> ⚠ Heavy artifacts (`checkpoints/`, `*.bin`, `*.pth`, `logs/`, `logits_cache/`,
> Cityscapes / Fishyscapes / SMIYC under `data/`) are git-ignored. See
> `.gitignore`.

---

## Setup

```bash
git clone <repo-url> && cd OASIS-Project
conda env create -f environment.yml && conda activate oasis
# or:  pip install -r requirements.txt
```

### Datasets

| Dataset | Used for | Download |
|---|---|---|
| Cityscapes (`leftImg8bit` + `gtFine`) | Steps 4, 5 | <https://www.cityscapes-dataset.com/> |
| Fishyscapes Lost & Found, Static | Steps 7, 8 | <https://fishyscapes.com/> |
| SegmentMeIfYouCan (RA-21, RO-21) | Steps 7, 8 | <https://segmentmeifyoucan.com/> |
| Road Anomaly | Steps 7, 8 | <https://www.epfl.ch/labs/cvlab/data/road-anomaly/> |

Edit the paths under `dataset:` in `configs/anomaly_eval.yaml` to match your
local layout (defaults assume `./data/...`).

### Pre-trained checkpoints

The two reference EoMT checkpoints (`eomt_coco.bin`, `eomt_cityscapes.bin`)
and the ERFNet weights come from the course Drive folder linked in the project
brief. Place them under `checkpoints/` — that folder is git-ignored.

For Step 5 you also need an *adapted* version of the COCO checkpoint that
matches the Cityscapes shape (1024×1024 input, 100 queries, 19 classes):

```bash
python scripts/adapt_coco_checkpoint.py \
    --src checkpoints/eomt_coco.bin \
    --dst checkpoints/eomt_coco_adapted.pth
```

---

## Usage

### Step 4 — mIoU baseline on Cityscapes-val

```bash
# EoMT pre-trained on Cityscapes (semantic)
python evaluate_miou.py --model eomt \
    --checkpoint checkpoints/eomt_cityscapes.bin

# EoMT pre-trained on COCO (panoptic) — predictions remapped to the
# 19 Cityscapes classes via the LUT in data/cityscapes.py
python evaluate_miou.py --model eomt --is-coco \
    --checkpoint checkpoints/eomt_coco.bin

# ERFNet baseline
python evaluate_miou.py --model erfnet \
    --checkpoint checkpoints/erfnet_pretrained.pth
```

### Step 5 — Fine-tune EoMT-COCO on Cityscapes-semantic

Training uses the Lightning stack in `training/`. Phase 1 fine-tunes only the
prediction head; phase 2 progressively unfreezes deeper layers.

```bash
cd training

# Phase 1 — frozen backbone (head only)
python main.py fit \
    --config configs/cityscapes_semantic.yaml \
    --config configs/phase1.yaml

# Phase 2 — resume and unfreeze
python main.py fit \
    --config configs/cityscapes_semantic.yaml \
    --config configs/phase2.yaml \
    --ckpt_path <path-to-phase1-last.ckpt>
```

The configs enable AMP (`trainer.precision: 16-mixed`), gradient accumulation,
and a two-stage warmup + polynomial decay schedule.

### Steps 7 & 8 — Anomaly segmentation

```bash
# ERFNet + MSP / MaxLogit / MaxEntropy
python evaluate_anomaly.py --model erfnet \
    --method msp --dataset fishyscapes \
    --config configs/anomaly_eval.yaml

# EoMT + RbA (the method that requires a mask architecture)
python evaluate_anomaly.py --model eomt \
    --method rba --dataset smiyc_anomaly \
    --config configs/anomaly_eval.yaml

# Temperature scaling — sweep over a grid on cached logits
python evaluate_anomaly.py --model eomt --method msp \
    --dataset fishyscapes --temperature-search \
    --config configs/anomaly_eval.yaml
```

Step 8 must be repeated for **three** EoMT checkpoints: COCO-pretrained,
Cityscapes-pretrained, and the fine-tuned one from Step 5. Select with
`--checkpoint` (or change `checkpoints.eomt_*` in `configs/anomaly_eval.yaml`).

---

## Post-hoc anomaly methods

| Method | Applies to | Score |
|---|---|---|
| **MSP** | ERFNet, EoMT | `1 − max softmax(logits / T)` |
| **MaxLogit** | ERFNet, EoMT | `−max logits` |
| **MaxEntropy** | ERFNet, EoMT | `H(softmax(logits / T))` |
| **RbA** | EoMT only | `−Σ_q tanh(mask_logit_q)` — pixel rejected by every query |

Temperature scaling (`T ∈ {0.5, 0.75, 1.0, 1.1, …}`) is searched on cached
logits, so the model forward pass runs once per (dataset, checkpoint) pair
(see `logits_cache.use_cache` in `configs/anomaly_eval.yaml`).

---

## Metrics

* **mIoU** — Cityscapes 19 classes (Steps 4 and 5).
* **AuPRC**, **FPR95**, **AuROC** — pixel-level anomaly detection on
  SMIYC RA-21, SMIYC RO-21, FS Lost & Found, FS Static, Road Anomaly.

All implementations live in `utils/metrics.py`.

---

## References

1. Chan et al. — *SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation*, NeurIPS 2021
2. Blum et al. — *The Fishyscapes Benchmark*, IJCV 2021
3. Cheng et al. — *MaskFormer*, NeurIPS 2021
4. Cheng et al. — *Mask2Former*, CVPR 2022
5. Oquab et al. — *DINOv2*, 2023
6. Kerssies et al. — *Your ViT is Secretly an Image Segmentation Model* (**EoMT**), CVPR 2025
7. Nayal et al. — *RbA: Segmenting Unknown Regions Rejected by All*, ICCV 2023
8. Hendrycks et al. — *Scaling Out-of-Distribution Detection for Real-World Settings*, ICML 2022
9. Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
10. Romera et al. — *ERFNet*, IEEE T-ITS 2018
11. Kirillov et al. — *Panoptic Segmentation*, CVPR 2019
12. Cordts et al. — *The Cityscapes Dataset*, CVPR 2016
13. Lin et al. — *Microsoft COCO*, ECCV 2014

---

## Credits

Course project for *Advanced Machine Learning / Machine Learning and Deep
Learning* — VANDAL Lab, Politecnico di Torino. EoMT code adapted from
[tue-mps/eomt](https://github.com/tue-mps/eomt) (MIT License); ERFNet from
[Eromera/erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch).
