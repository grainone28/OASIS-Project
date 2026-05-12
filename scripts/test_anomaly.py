import torch
import numpy as np
from utils.anomaly_methods import msp_anomaly_score
from utils.metrics import evaluate_anomaly

print("--- INTEGRATION TEST: Anomaly Logic + Metrics ---")

# 1. Simulate Model Output (Logits)
# Batch 1, 19 Classes, 100x100 pixels
dummy_logits = torch.randn(1, 19, 100, 100)

# 2. Member 4 Action: Compute Anomaly Scores using MSP
# Returns a map of [1, 100, 100]
print("1. Extracting anomaly scores...")
anomaly_scores_tensor = msp_anomaly_score(dummy_logits, temperature=1.5)
anomaly_scores_numpy = anomaly_scores_tensor.numpy()

# 3. Simulate Ground Truth Labels
# 0 = normal road, 1 = anomaly, 255 = sky (ignore)
dummy_labels = np.random.choice([0, 1, 255], size=(1, 100, 100), p=[0.8, 0.1, 0.1])

# 4. Member 2 Action: Evaluate the scores
print("2. Calculating metrics...")
results = evaluate_anomaly(anomaly_scores_numpy, dummy_labels)

print("\n✅ INTEGRATION SUCCESSFUL!")
print(f"AuPRC: {results['AuPRC']:.4f}")
print(f"FPR95: {results['FPR95']:.4f}")
print(f"AuROC: {results['AuROC']:.4f}")