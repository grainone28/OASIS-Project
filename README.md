# OASIS — Out-of-distribution Analysis for Semantic Image Segmentation

*Comprehensive Road Scene Understanding for Autonomous
Driving* (Politecnico di Torino, VANDAL Lab).
The goal is to compare a pixel-based baseline (ERFNet) against a
modern mask-based architecture (EoMT, CVPR 2025) on two tasks: closed-set
semantic segmentation on Cityscapes, and pixel-level anomaly segmentation on
driving benchmarks (Fishyscapes Lost & Found / Static, SegmentMeIfYouCan
RoadAnomaly21 / RoadObstacle21, Road Anomaly).

Authors: Rainone Gerardo, Stasio Imma, D'Amico Davide, Di Foggia Alessandra.

## What this project implements

The repository covers the eight steps of the project brief through three
main implementation blocks:

1. **mIoU evaluation on Cityscapes-val** (`evaluate_miou.py`), comparing the
   EoMT checkpoint pre-trained on Cityscapes-semantic against the one
   pre-trained on COCO-panoptic. The comparison is non-trivial: the two
   models operate in different label spaces (19 vs. 133 classes), so COCO
   predictions are remapped before computing the metric through a lookup
   table defined in `data/cityscapes.py`.

2. **Fine-tuning of EoMT-COCO on Cityscapes-semantic** (`training/main.py`),
   to test whether the broader visual priors learned on COCO, once
   specialised on driving data, can compete with the model trained on
   Cityscapes from the start. Fine-tuning is organised in two phases (head
   only, then progressive unfreezing with layer-wise learning rate decay).

3. **Post-hoc anomaly segmentation** (`evaluate_anomaly.py`), applying MSP,
   MaxLogit and MaxEntropy to both architectures, RbA to EoMT only, and a
   temperature scaling grid search on all combinations.

## Design choices

### Two coexisting training-time stacks

`training/` is a vendored copy of the official
[tue-mps/eomt](https://github.com/tue-mps/eomt) repository. It is built on
PyTorch Lightning with a layered jsonargparse configuration system that is
non-trivial to replace. We chose not to rewrite it from scratch: the
original training pipeline already implements AMP, a two-stage warmup +
polynomial schedule, and the correct padding logic for masked attention,
and replacing it would have introduced bugs without any obvious benefit.

For evaluation and inference, however, only the model definitions
(`training/models/eomt.py`, `vit.py`, `erfnet.py`) are needed. We import
them directly from the top-level scripts, which avoids dragging Lightning
and its dependencies into the evaluation pipeline.

### Two-phase fine-tuning

The brief suggests starting with a frozen backbone and progressively
unfreezing the deeper layers. We split this into two separate config
files rather than relying on a single training loop with internal
scheduling:

- `phase1.yaml`: backbone frozen, only the prediction head is trained.
  This anchors the head to the new label space (19 instead of 133
  classes) without disturbing the DINOv2 features.
- `phase2.yaml`: resumes from the phase-1 checkpoint and unfreezes deeper
  layers using layer-wise learning rate decay (`llrd`).

Keeping the two phases in separate configs makes it straightforward to
resume only phase 2 from a saved checkpoint, which matters when training
on Colab and the runtime expires mid-experiment.

### COCO checkpoint adaptation

The COCO checkpoint was trained at 640×640 with 200 queries on 133
panoptic classes. Cityscapes uses 1024×1024, 100 queries, and 19 classes.
Loading the weights as-is fails because `pos_embed`, `q.weight` and
`class_head` all have incompatible shapes. `scripts/adapt_coco_checkpoint.py`
performs three operations:

1. bicubic interpolation of `pos_embed` onto the new patch grid,
2. truncation of the excess query embeddings,
3. removal of `class_head` (re-initialised from scratch).

What remains is enough to start fine-tuning from informed weights rather
than from random initialisation.

### RbA restricted to EoMT

MSP, MaxLogit and MaxEntropy are defined pixel-wise on the class logits
and apply to any classifier. RbA (*Rejected by All*, Nayal et al., ICCV
2023) instead exploits the mask-based structure: it aggregates the mask
logits of all queries through `tanh` and flags as anomalous those pixels
that no query claims with high confidence. It therefore only makes sense
on EoMT, where the query mechanism exists.

### Cached logits for temperature search

A grid search over the temperature `T` (`--temperature-search`) would be
prohibitively slow if every value required a forward pass through the
model. To avoid this, the first time a `(checkpoint, dataset)` pair is
evaluated we serialise the raw logits to disk; all subsequent temperature
values are then computed on the cached tensors. This is controlled by
`logits_cache.use_cache` in `configs/anomaly_eval.yaml`.

## Repository layout

```
OASIS-Project/
├── configs/      # YAML configuration for the evaluation scripts
├── data/         # Cityscapes and OoD dataset loaders, transforms, label LUTs
├── scripts/      # one-shot utilities (e.g. COCO checkpoint adaptation)
├── training/     # Lightning training stack vendored from tue-mps/eomt
│   ├── configs/  # training configs (Cityscapes base + phase1 / phase2)
│   ├── datasets/ # Lightning data modules
│   ├── models/   # EoMT, DINOv2-based ViT, ERFNet
│   └── train/    # Lightning module, loss, scheduler
├── utils/        # anomaly scores, metrics, post-processing, visualisation
├── evaluate_miou.py     # entry point for the closed-set evaluation
└── evaluate_anomaly.py  # entry point for the anomaly evaluation
```

Checkpoints, datasets, training logs and the logits cache are excluded
from version control (see `.gitignore`).

## Setup

```bash
git clone <repo-url> && cd OASIS-Project
conda env create -f environment.yml && conda activate oasis
```

The repository expects three pre-trained checkpoints under `checkpoints/`:
the ERFNet weights, EoMT trained on Cityscapes-semantic, and EoMT trained
on COCO-panoptic. Before launching the fine-tuning, the COCO checkpoint
must be adapted to the Cityscapes geometry:

```bash
python scripts/adapt_coco_checkpoint.py \
    --src checkpoints/eomt_coco.bin \
    --dst checkpoints/eomt_coco_adapted.pth
```

Dataset paths must be set in `configs/anomaly_eval.yaml` (Cityscapes,
Fishyscapes L&F and Static, SMIYC RoadAnomaly21 and RoadObstacle21, Road
Anomaly).

## Running the experiments

Closed-set semantic segmentation on Cityscapes-val:

```bash
python evaluate_miou.py --model eomt --checkpoint checkpoints/eomt_cityscapes.bin
python evaluate_miou.py --model eomt --checkpoint checkpoints/eomt_coco.bin --is-coco
python evaluate_miou.py --model erfnet --checkpoint checkpoints/erfnet_pretrained.pth
```

Fine-tuning (must be launched from inside `training/`):

```bash
cd training
python main.py fit --config configs/cityscapes_semantic.yaml --config configs/phase1.yaml
python main.py fit --config configs/cityscapes_semantic.yaml --config configs/phase2.yaml \
    --ckpt_path lightning_logs_phase1/.../last.ckpt
```

Anomaly segmentation:

```bash
# pixel baseline
python evaluate_anomaly.py --model erfnet --method msp --dataset fishyscapes \
    --config configs/anomaly_eval.yaml

# mask baseline with RbA
python evaluate_anomaly.py --model eomt --method rba --dataset smiyc_anomaly \
    --config configs/anomaly_eval.yaml

# temperature scaling sweep on cached logits
python evaluate_anomaly.py --model eomt --method msp --dataset fishyscapes \
    --temperature-search --config configs/anomaly_eval.yaml
```

The anomaly evaluation must be repeated for each of the three EoMT
checkpoints (COCO, Cityscapes, fine-tuned) over all five anomaly
benchmarks to populate the full results table.

## Results

### Closed-set semantic segmentation (Cityscapes-val, mIoU)

| Model                       | mIoU (%) |
|-----------------------------|----------|
| ERFNet                      | 71.99    |
| EoMT — pre-trained on COCO  | 50.02    |
| EoMT — pre-trained on CS    | 77.73    |
| EoMT — fine-tuned           | 70.88    |

### Anomaly segmentation baselines

Pixel-level metrics on the five OoD benchmarks **(AuPRC/FPR95)**. AuPRC and FPR95 are
reported per dataset; mIoU refers to the closed-set performance of each
backbone.

| Model  | mIoU  | Method      | SMIYC RA-21     | SMIYC RO-21   | FS L&F        | FS Static     | Road Anomaly    |
|--------|-------|-------------|-----------------|---------------|---------------|---------------|-----------------|
| ERFNet | 71.99 | MSP         | 0.292 / 0.624   | 0.028 / 0.646 | 0.017 / 0.503 | 0.050 / 0.409 | 0.124 / 0.825   |
| ERFNet |       | MaxLogit    | 0.384 / 0.593   | 0.047 / 0.483 | 0.033 / 0.451 | 0.062 / 0.397 | 0.156 / 0.733   |
| ERFNet |       | MaxEntropy  | 0.311 / 0.625   | 0.032 / 0.653 | 0.025 / 0.500 | 0.058 / 0.406 | 0.127 / 0.826   |
| EoMT   | 70.88 | MSP         | 0.527 / 0.909   | 0.329 / 1.000 | 0.294 / 0.908 | 0.515 / 0.949 | 0.414 / 0.958   |
| EoMT   |       | MaxLogit    | 0.527 / 0.909   | 0.329 / 1.000 | 0.294 / 0.908 | 0.515 / 0.949 | 0.414 / 0.958   |
| EoMT   |       | MaxEntropy  | 0.337 / 0.797   |*0.807 / 0.017*| 0.016 / 0.399 | 0.123 / 0.484 | 0.579 / 0.642   |
| EoMT   |       | RbA         | 0.618* / 0.272 | 0.491 / 0.216 |*0.307* / 0.202|*0.558* / 0.152| 0.487 / 0.273   |

The anomaly table is reported for each of the three EoMT checkpoints
(COCO, Cityscapes, fine-tuned).

### Temperature scaling on MSP

| Method        | mIoU  | SMIYC RA-21       | SMIYC RO-21   | FS L&F        | FS Static     | Road Anomaly  |
|---------------|-------|-------------------|---------------|---------------|---------------|---------------|
| MSP (T=1.0)   | 70.88 | 0.527 / 0.909     | 0.329 / 1.000 | 0.294 / 0.908 | 0.515 / 0.949 | 0.414 / 0.958 |
| MSP (T=0.5)   |       | 0.242 / 0.945     | 0.200 / 1.000 | 0.267 / 0.939 | 0.475 / 0.965 | 0.328 / 0.971 |
| MSP (best=2.0)|       | *0.699 / 0.224*   |*0.353 / 1.000*|*0.339 / 0.861*|*0.564 / 0.206*|*0.455 / 0.934*|


## Notes

`training/main.py` imports `from datasets.lightning_data_module import ...`
without the `training.` prefix, so it must be executed from inside the
`training/` directory. This behaviour is inherited from the original
tue-mps repository; we did not modify it to preserve compatibility with
the upstream configs.

## References

- Kerssies et al., *Your ViT is Secretly an Image Segmentation Model*
  (EoMT), CVPR 2025.
- Nayal et al., *RbA: Segmenting Unknown Regions Rejected by All*,
  ICCV 2023.
- Romera et al., *ERFNet: Efficient Residual Factorized ConvNet for
  Real-Time Semantic Segmentation*, IEEE T-ITS 2018.
- Oquab et al., *DINOv2: Learning Robust Visual Features without
  Supervision*, 2023.
- Cheng et al., *Masked-attention Mask Transformer for Universal Image
  Segmentation* (Mask2Former), CVPR 2022.
- Chan et al., *SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation*,
  NeurIPS 2021.
- Blum et al., *The Fishyscapes Benchmark: Anomaly Detection for Semantic
  Segmentation*, IJCV 2021.
- Hendrycks et al., *Scaling Out-of-Distribution Detection for Real-World
  Settings*, ICML 2022.
- Kirillov et al., *Panoptic Segmentation*, CVPR 2019.
- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene
  Understanding*, CVPR 2016.

Datasets and pre-trained checkpoints used in this work were provided by
the course instructors (VANDAL Lab, Politecnico di Torino). EoMT code is
adapted from [tue-mps/eomt](https://github.com/tue-mps/eomt) under the
MIT License; ERFNet from
[Eromera/erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch).
