# OASIS — Out-of-distribution Analysis for Semantic Image Segmentation

*Comprehensive Road Scene Understanding for Autonomous
Driving* (Politecnico di Torino, VANDAL Lab).

The goal is to compare a pixel-based baseline (ERFNet) against a
modern mask-based architecture (EoMT, CVPR 2025) on two tasks: closed-set
semantic segmentation on Cityscapes, and pixel-level anomaly segmentation on
driving benchmarks (Fishyscapes Lost & Found / Static, SegmentMeIfYouCan
RoadAnomaly21 / RoadObstacle21, Road Anomaly).

**Authors: Rainone Gerardo, Stasio Imma, D'Amico Davide, Di Foggia Alessandra.**



## What this project implements

Three main implementation blocks:

1. **mIoU evaluation on Cityscapes** (`evaluate_miou.py`), comparing the
   EoMT checkpoint pre-trained on Cityscapes-semantic against the one
   pre-trained on COCO-panoptic. The comparison is non-trivial: the two
   models operate in different label spaces (19 vs. 133 classes), so COCO
   predictions are remapped before computing the metric through a lookup
   table defined in `data/cityscapes.py`.

2. **Fine-tuning of EoMT-COCO on Cityscapes-semantic** (`training/main.py`),
   to test whether the broader visual priors learned on COCO, once
   specialised on driving data, can compete with the model trained on
   Cityscapes from the start. Fine-tuning is organised in two phases: head
   only, then progressive unfreezing with layer-wise learning rate decay.

3. **Post-hoc anomaly segmentation** (`evaluate_anomaly.py`), applying MSP,
   MaxLogit and MaxEntropy to both architectures, RbA to EoMT only, and a
   temperature scaling grid search on all combinations.

## Design choices

### Two coexisting training-time stacks

The repository contains two distinct pipelines that coexist without interfering with each other.

The first is a full training stack located in training/, built on PyTorch Lightning with a layered 
jsonargparse configuration system. This pipeline handles AMP, a two-stage learning rate schedule 
(head-only warm-up followed by full backbone fine-tuning with layer-wise learning rate decay), 
and the masked attention annealing logic required by EoMT. It is invoked exclusively through 
main.py and is not imported anywhere else. 

The second is a lightweight evaluation stack built
on top of the top-level scripts (evaluate_miou.py, evaluate_anomaly.py, train.py). 
These scripts import only the model definitions directly from training/models/ — specifically eomt.py, 
vit.py, and scale_block.py — without pulling in Lightning or any of its dependencies. 
This keeps the evaluation pipeline fast, dependency-free, and easy to inspect.
The separation is intentional: training correctness requires the full Lightning machinery, 
while evaluation and inference require only PyTorch and the model weights.


### Two-phase fine-tuning

The brief suggests starting with a frozen backbone and progressively
unfreezing. We split this into two configuration files,
`training/configs/phase1.yaml` and `phase2.yaml`, which differ in five
points:

- `ckpt_path`: phase 1 starts from the adapted COCO checkpoint, phase 2
  resumes from the last checkpoint produced by phase 1.
- `load_ckpt_class_head`: `false` in phase 1 (classification head is
  reinitialised on the 19 Cityscapes classes), `true` in phase 2.
- `llrd` (layer-wise learning rate decay): `0.0` in phase 1, `0.9` in
  phase 2.
- `lr_mult`: `0.0` in phase 1, `0.1` in phase 2.
- `attn_mask_annealing_enabled`: `false` in phase 1, `true` in phase 2.

Both phases are configured for 10 epochs. Keeping them in separate
configs makes it straightforward to resume only phase 2 from a saved
checkpoint, which matters when training on Colab and the runtime
expires mid-experiment.



### COCO checkpoint adaptation

The COCO checkpoint was trained at 640×640 with 200 queries on 133
panoptic classes. Our fine-tuning on Cityscapes uses 1024×1024 inputs
(set in `training/datasets/cityscapes_semantic.py`, `img_size=(1024,
1024)`), 100 queries and 19 classes. Loading the COCO weights as-is
fails because `pos_embed`, `q.weight` and `class_head` all have
incompatible shapes. `scripts/adapt_coco_checkpoint.py` performs three
operations:

1. bicubic interpolation of `pos_embed` from the source 40×40 patch
   grid to the target 64×64 grid (controlled by `--target-grid`),
2. truncation of the query embeddings from 200 to 100 (controlled by
   `--target-num-q`),
3. removal of `class_head` (re-initialised from scratch on the 19
   Cityscapes classes).

What remains is enough to start fine-tuning from informed weights rather
than from random initialisation.

### RbA restricted to EoMT

MSP, MaxLogit and MaxEntropy are defined pixel-wise on the class logits
and apply to any classifier. RbA (*Rejected by All*) instead exploits the mask-based structure: 
it combines, for each query, the sigmoid of the mask logit with the softmax confidence on 
known classes, and flags as anomalous those pixels for which no query reaches a high combined score. 
It therefore only makes sense on EoMT, where the query mechanism exists.

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
# pixel baseline (ERFNet + MSP) on Fishyscapes Lost & Found
python evaluate_anomaly.py --model erfnet --method msp --dataset fs_laf \
    --config configs/anomaly_eval.yaml

# mask baseline with RbA on SMIYC RoadAnomaly21 (fine-tuned EoMT checkpoint)
python evaluate_anomaly.py --model eomt --eomt-version finetuned \
    --method rba --dataset smiyc_ra21 \
    --config configs/anomaly_eval.yaml

# temperature scaling sweep on cached logits (EoMT + MSP on FS L&F)
python evaluate_anomaly.py --model eomt --eomt-version finetuned \
    --method msp --dataset fs_laf --temperature-search \
    --config configs/anomaly_eval.yaml
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

| Model  | mIoU  | Method      | SMIYC RA-21     | SMIYC RO-21     | FS L&F          | FS Static       | Road Anomaly    |
|--------|-------|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| ERFNet | 71.99 | MSP         | 0.292 / 0.624   | 0.028 / 0.646   | 0.017 / 0.503   | 0.050 / 0.409   | 0.124 / 0.825   |
| ERFNet |       | MaxLogit    | 0.384 / 0.593   | 0.047 / 0.483   | 0.033 / 0.451   | 0.062 / 0.397   | 0.156 / 0.733   |
| ERFNet |       | MaxEntropy  | 0.311 / 0.625   | 0.032 / 0.653   | 0.025 / 0.500   | 0.058 / 0.406   | 0.127 / 0.826   |
| EoMT   | 70.88 | MSP         | 0.527 / 0.909   | 0.329 / 1.000   | 0.294 / 0.908   | 0.515 / 0.949   | 0.414 / 0.958   |
| EoMT   |       | MaxLogit    | 0.527 / 0.909   | 0.329 / 1.000   | 0.294 / 0.908   | 0.515 / 0.949   | 0.414 / 0.958   |  
| EoMT   |       | MaxEntropy  | 0.337 / 0.797   | 0.807* / 0.017* | 0.016 / 0.399   | 0.123 / 0.484   | 0.579* / 0.642  |
| EoMT   |       | RbA         | 0.618* / 0.272* | 0.491 / 0.216   | 0.307* / 0.202* | 0.558* / 0.152* | 0.487 / 0.273*  |


### Temperature scaling on MSP

| Method        | mIoU  | SMIYC RA-21       | SMIYC RO-21     | FS L&F          | FS Static       | Road Anomaly    |
|---------------|-------|-------------------|-----------------|-----------------|-----------------|-----------------|
| MSP (T=1.0)   | 70.88 | 0.527 / 0.909     | 0.329 / 1.000   | 0.294 / 0.908   | 0.515 / 0.949   | 0.414 / 0.958   |
| MSP (T=0.5)   |       | 0.242 / 0.945     | 0.200 / 1.000   | 0.267 / 0.939   | 0.475 / 0.965   | 0.328 / 0.971   |
| MSP (best=2.0)|       | 0.699* / 0.224*   | 0.353* / 1.000* | 0.339* / 0.861* | 0.564* / 0.206* | 0.455* / 0.934* |



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
the course instructors (VANDAL Lab, Politecnico di Torino). 

Source: https://github.com/AlessandroMarinai/MaskArchitectureAnomaly_CourseProject/tree/main
