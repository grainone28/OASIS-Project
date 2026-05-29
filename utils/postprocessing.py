import torch
import torch.nn.functional as F


def queries_to_segmentation(pred_masks, pred_logits, threshold=0.5, method='max'):
    B, Q, H, W = pred_masks.shape
    _, _, C = pred_logits.shape  
    
    device = pred_masks.device
    
    query_class_scores, query_class_ids = pred_logits.max(dim=-1)  
    
    segmentation_logits = torch.zeros(B, C, H, W, device=device)
    
    if method == 'max':
        max_scores = torch.full((B, H, W), -1e6, device=device)
        pred_classes = torch.zeros(B, H, W, dtype=torch.long, device=device)
        
        for b in range(B):
            for q in range(Q):
                mask = pred_masks[b, q] > threshold  
                class_id = query_class_ids[b, q].item()
                score = query_class_scores[b, q].item()
                
                update_mask = (mask) & (score > max_scores[b])
                pred_classes[b][update_mask] = class_id
                max_scores[b][update_mask] = score
                segmentation_logits[b, class_id][update_mask] = score
    
    elif method == 'accumulate':
        for b in range(B):
            for q in range(Q):
                mask = pred_masks[b, q] > threshold
                class_id = query_class_ids[b, q].item()
                score = query_class_scores[b, q]
                segmentation_logits[b, class_id] += mask.float() * score
        
        pred_classes = segmentation_logits.argmax(dim=1)  
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return segmentation_logits, pred_classes


def queries_to_segmentation_fast(pred_masks, pred_logits, threshold=0.5):
    mask_cls  = torch.softmax(pred_logits[..., :-1], dim=-1).float()  
    mask_pred = torch.sigmoid(pred_masks).float()                      
    sem_map   = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)  
    return sem_map.argmax(dim=1)                               

if __name__ == "__main__":
    print("Testing postprocessing functions...")
    
    B, Q, H, W, C = 2, 100, 512, 1024, 20
    pred_masks = torch.sigmoid(torch.randn(B, Q, H, W))  
    pred_logits = torch.randn(B, Q, C)
    
    print(f"Input shapes:")
    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  pred_logits: {pred_logits.shape}")
    
    seg_logits, pred_classes = queries_to_segmentation(pred_masks, pred_logits)
    print(f"\nOutput (slow method):")
    print(f"  segmentation_logits: {seg_logits.shape}")
    print(f"  pred_classes: {pred_classes.shape}")
    print(f"  Unique classes predicted: {torch.unique(pred_classes)}")
    
    pred_classes_fast = queries_to_segmentation_fast(pred_masks, pred_logits)
    print(f"\nOutput (fast method):")
    print(f"  pred_classes: {pred_classes_fast.shape}")
    print(f"  Unique classes predicted: {torch.unique(pred_classes_fast)}")
    
    print("\n Postprocessing module is ready!")