import torch
import numpy as np
from utils.anomaly_methods import msp_anomaly_score
from utils.metrics import evaluate_anomaly

dummy_logits = torch.randn(1, 19, 100, 100)


anomaly_scores_tensor = msp_anomaly_score(dummy_logits, temperature=1.5)
anomaly_scores_numpy = anomaly_scores_tensor.numpy()


dummy_labels = np.random.choice([0, 1, 255], size=(1, 100, 100), p=[0.8, 0.1, 0.1])

results = evaluate_anomaly(anomaly_scores_numpy, dummy_labels)

print(f"AuPRC: {results['AuPRC']:.4f}")
print(f"FPR95: {results['FPR95']:.4f}")
print(f"AuROC: {results['AuROC']:.4f}")