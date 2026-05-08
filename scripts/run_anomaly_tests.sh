#!/usr/bin/env bash
# scripts/run_anomaly_tests.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs all anomaly segmentation evaluations (Steps 6, 7, 8).
# Run from project root:  bash scripts/run_anomaly_tests.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG="configs/anomaly_eval.yaml"

echo "=============================================="
echo " VANDAL — Anomaly Evaluation Suite"
echo "=============================================="

# ── Step 6 & 7: ERFNet + MSP on Fishyscapes ──────────────────────────────────
echo ""
echo "[Step 6/7] ERFNet + MSP — Fishyscapes Lost & Found"
python evaluate_anomaly.py \
    --config "$CONFIG" \
    --method msp \
    --model  erfnet \
    --dataset fishyscapes \
    --temperature 1.0

# ── Step 7: ERFNet + MSP on SMIYC ─────────────────────────────────────────────
echo ""
echo "[Step 7] ERFNet + MSP — SMIYC RoadAnomaly21"
python evaluate_anomaly.py \
    --config "$CONFIG" \
    --method msp \
    --model  erfnet \
    --dataset smiyc_anomaly \
    --temperature 1.0

# ── Step 8: EoMT + RbA on Fishyscapes ────────────────────────────────────────
echo ""
echo "[Step 8] EoMT + RbA — Fishyscapes (default T=1.0)"
python evaluate_anomaly.py \
    --config "$CONFIG" \
    --method rba \
    --model  eomt \
    --dataset fishyscapes \
    --temperature 1.0

# ── Step 8: Temperature scaling grid search ────────────────────────────────────
echo ""
echo "[Step 8] EoMT + RbA — Temperature grid search on Fishyscapes"
python evaluate_anomaly.py \
    --config "$CONFIG" \
    --method rba \
    --model  eomt \
    --dataset fishyscapes \
    --temperature-search

echo ""
echo "All anomaly evaluations complete."
