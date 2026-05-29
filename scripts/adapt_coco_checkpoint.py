import argparse, torch
import torch.nn.functional as F

def adapt(src, dst, target_grid=64, target_num_q=100):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    for k in list(state.keys()):
        if "pos_embed" in k:
            pe = state[k]
            src_h = int(pe.shape[1] ** 0.5)
            if pe.shape[1] == target_grid * target_grid: continue
            pe = pe.reshape(1, src_h, src_h, -1).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
            state[k] = pe.permute(0, 2, 3, 1).reshape(1, target_grid**2, -1)
        elif k.endswith("q.weight") and state[k].shape[0] != target_num_q:
            state[k] = state[k][:target_num_q]
        elif any(s in k for s in ("class_head", "class_predictor", "criterion.empty_weight")):
            del state[k]

    torch.save({"state_dict": state}, dst)
    print(f"✓ Saved {dst} ({len(state)} keys)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--target-grid", type=int, default=64)
    p.add_argument("--target-num-q", type=int, default=100)
    a = p.parse_args()
    adapt(a.src, a.dst, a.target_grid, a.target_num_q)
    