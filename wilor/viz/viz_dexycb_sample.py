"""
DexYCB test split에서 샘플 하나를 골라 WiLoR로 inference하고,
GT 3D pose와 pred 3D pose(절대 translation solver 적용)를 JSON으로 저장.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset
from solve_absolute_depth import solve_absolute_translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz")
    parser.add_argument("--img_dir", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--checkpoint", default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--sample_idx", type=int, default=0, help="test split 내 인덱스 (0부터)")
    parser.add_argument("--pelvis_id", type=int, default=0)
    parser.add_argument("--out", default="/workspace/drag1008/wilor/viz_sample.json")
    args = parser.parse_args()

    model, cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    val_dataset_full = ImageDataset(cfg, args.dataset_npz, args.img_dir, train=False, rescale_factor=-1)
    official = val_dataset_full.data["official_split"]
    test_idx = np.where(official == "test")[0]
    print(f"test split size: {len(test_idx)}")

    chosen = test_idx[args.sample_idx: args.sample_idx + 1]
    loader = DataLoader(Subset(val_dataset_full, chosen), batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        return x

    batch = to_device(batch)

    with torch.no_grad():
        output = model.forward_step(batch, train=False)

    pred_keypoints_3d = output["pred_keypoints_3d"]  # (1,21,3)
    rel_3d = pred_keypoints_3d - pred_keypoints_3d[:, args.pelvis_id:args.pelvis_id + 1, :]
    pred_keypoints_2d = output["pred_keypoints_2d"]

    t_pred = solve_absolute_translation(
        pred_keypoints_2d, rel_3d,
        batch["box_center"].float(), batch["box_size"].float(),
        batch["fx"].float(), batch["fy"].float(), batch["px"].float(), batch["py"].float(),
    )
    pred_abs = (rel_3d + t_pred.unsqueeze(1))[0].cpu().numpy()  # (21,3) meters

    gt_abs = batch["keypoints_3d"][0, :, :3].cpu().numpy()  # (21,3) meters

    # ---- 2D: network's direct 2D prediction, denormalized to original image pixels ----
    center = batch["box_center"].float().view(1, 1, 2)
    b = batch["box_size"].float().view(1, 1, 1)
    pred_2d_pixels = (center + pred_keypoints_2d * b)[0].cpu().numpy()  # (21,2)

    gt_2d_pixels = batch["orig_keypoints_2d"][0, :, :2].cpu().numpy()  # (21,2)

    imgname = str(val_dataset_full.imgname[chosen[0]])

    out = {
        "imgname": imgname,
        "pred_joints_m": pred_abs.tolist(),
        "gt_joints_m": gt_abs.tolist(),
        "pred_2d_px": pred_2d_pixels.tolist(),
        "gt_2d_px": gt_2d_pixels.tolist(),
        "right": float(val_dataset_full.right[chosen[0]]),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"saved to {args.out}")
    print(f"imgname: {imgname}")
    pa_err = np.linalg.norm(pred_abs - pred_abs[args.pelvis_id] - (gt_abs - gt_abs[args.pelvis_id]), axis=-1).mean() * 1000
    print(f"pelvis-relative MPJPE (no procrustes): {pa_err:.2f} mm")
    err2d = np.linalg.norm(pred_2d_pixels - gt_2d_pixels, axis=-1).mean()
    print(f"2D pixel MPJPE: {err2d:.2f} px")


if __name__ == "__main__":
    main()
