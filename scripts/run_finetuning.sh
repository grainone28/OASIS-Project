#!/usr/bin/env bash
# scripts/run_finetuning.sh
# ─────────────────────────────────────────────────────────────────────────────
# Reproducible fine-tuning script for EoMT on Cityscapes.
# Run from the project root:  bash scripts/run_finetuning.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG="configs/default_eomt.yaml"
LOGDIR="logs/finetuning_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo " VANDAL — EoMT Fine-Tuning"
echo " Config  : $CONFIG"
echo " Log dir : $LOGDIR"
echo "=============================================="

# Sanity check: CUDA available?
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

mkdir -p "$LOGDIR"

python train.py --config "$CONFIG" 2>&1 | tee "$LOGDIR/train.log"

echo ""
echo "Training complete. Checkpoints saved in: checkpoints/"
echo "Logs saved in: $LOGDIR/train.log"
