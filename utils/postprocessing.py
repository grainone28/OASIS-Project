"""
Post-processing utilities for EoMT model outputs.
Converts DETR-style query predictions to dense segmentation maps

Owner: Membro 3 (Model Architect)
Usage: Both Membro 2 (Evaluator) and Membro 3 can import from this file.
"""

import torch
import torch.nn.functional as F


def queries_to_segmentation(pred_masks, pred_logits, threshold=0.5, method='max'):
    """
    Converts DETR-style query predictions to dense segmentation maps.
    
    Args:
        pred_masks: Tensor [B, Q, H, W] - Binary masks for each query
        pred_logits: Tensor [B, Q, C] - Class logits for each query (C=20 for Cityscapes)
        threshold: Float - Confidence threshold for mask binarization
        method: str - 'max' (take highest scoring class per pixel) or 'accumulate'
    
    Returns:
        segmentation: Tensor [B, C, H, W] - Dense segmentation logits
        pred_classes: Tensor [B, H, W] - Final predicted class per pixel (for mIoU)
    """
    B, Q, H, W = pred_masks.shape
    _, _, C = pred_logits.shape  # C = 20 classes
    
    device = pred_masks.device
    
    # Get predicted class for each query
    query_class_scores, query_class_ids = pred_logits.max(dim=-1)  # [B, Q]
    
    # Initialize output
    segmentation_logits = torch.zeros(B, C, H, W, device=device)
    
    if method == 'max':
        # For each pixel, keep only the highest-scoring query
        max_scores = torch.full((B, H, W), -1e6, device=device)
        pred_classes = torch.zeros(B, H, W, dtype=torch.long, device=device)
        
        for b in range(B):
            for q in range(Q):
                mask = pred_masks[b, q] > threshold  # [H, W]
                class_id = query_class_ids[b, q].item()
                score = query_class_scores[b, q].item()
                
                # Update pixels where this query has higher score
                update_mask = (mask) & (score > max_scores[b])
                pred_classes[b][update_mask] = class_id
                max_scores[b][update_mask] = score
                segmentation_logits[b, class_id][update_mask] = score
    
    elif method == 'accumulate':
        # Sum all query contributions per class
        for b in range(B):
            for q in range(Q):
                mask = pred_masks[b, q] > threshold
                class_id = query_class_ids[b, q].item()
                score = query_class_scores[b, q]
                segmentation_logits[b, class_id] += mask.float() * score
        
        pred_classes = segmentation_logits.argmax(dim=1)  # [B, H, W]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return segmentation_logits, pred_classes


def queries_to_segmentation_fast(pred_masks, pred_logits, threshold=0.5):
    """
    Faster vectorized version using torch operations.g
    
    Args:
        pred_masks: Tensor [B, Q, H, W]
        pred_logits: Tensor [B, Q, C]
        threshold: Float
    
    Returns:
        pred_classes: Tensor [B, H, W] - Ready for mIoU calculation
    """
    B, Q, H, W = pred_masks.shape
    _, _, C = pred_logits.shape
    
    # Get class predictions and scores
    scores, class_ids = pred_logits.max(dim=-1)  # [B, Q]
    
    # Binarize masks
    binary_masks = (pred_masks > threshold).float()  # [B, Q, H, W]
    
    # Weight masks by their classification score
    weighted_masks = binary_masks * scores.unsqueeze(-1).unsqueeze(-1)  # [B, Q, H, W]
    
    # For each pixel, find the query with highest score
    best_query_idx = weighted_masks.argmax(dim=1)  # [B, H, W]
    
    # Map query indices to class IDs
    # Expand best_query_idx to match class_ids dimensions for gathering
    batch_idx = torch.arange(B, device=class_ids.device).view(B, 1, 1).expand(B, H, W)
    pred_classes = class_ids[batch_idx, best_query_idx]  # [B, H, W]
    
    return pred_classes


# Test function
if __name__ == "__main__":
    print("Testing postprocessing functions...")
    
    # Simulate EoMT outputs
    B, Q, H, W, C = 2, 100, 512, 1024, 20
    pred_masks = torch.sigmoid(torch.randn(B, Q, H, W))  # [0, 1] range
    pred_logits = torch.randn(B, Q, C)
    
    print(f"Input shapes:")
    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  pred_logits: {pred_logits.shape}")
    
    # Test slow version
    seg_logits, pred_classes = queries_to_segmentation(pred_masks, pred_logits)
    print(f"\nOutput (slow method):")
    print(f"  segmentation_logits: {seg_logits.shape}")
    print(f"  pred_classes: {pred_classes.shape}")
    print(f"  Unique classes predicted: {torch.unique(pred_classes)}")
    
    # Test fast version
    pred_classes_fast = queries_to_segmentation_fast(pred_masks, pred_logits)
    print(f"\nOutput (fast method):")
    print(f"  pred_classes: {pred_classes_fast.shape}")
    print(f"  Unique classes predicted: {torch.unique(pred_classes_fast)}")
    
    print("\n Postprocessing module is ready!")