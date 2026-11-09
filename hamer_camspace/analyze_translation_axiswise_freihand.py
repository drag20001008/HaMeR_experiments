"""
FreiHAND: wrist(root) translation을 두 가지 방식(Model-Predicted / PnP)으로 복원했을 때,
GT wrist 절대좌표와의 오차를 X/Y/Z 축별로 분리해서 계산.
전체 21개 관절 MPJPE가 아니라 "translation(=root) 자체"만 따로 봄.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    build_dataset as build_dataset_base, build_cfg, build_model, solve_translation,
)


def axis_stats(err):
    # err: (N,3) in meters, signed (pred - gt)
    err_mm = err * 1000.0
    mae = np.abs(err_mm).mean(axis=0)       # (3,) mean absolute error per axis
    bias = err_mm.mean(axis=0)              # (3,) mean signed error (systematic drift)
    rmse = np.sqrt((err_mm ** 2).mean(axis=0))
    return {
        "mae_xyz_mm": mae.tolist(),
        "bias_xyz_mm": bias.tolist(),
        "rmse_xyz_mm": rmse.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_dir, use_beta_adapter=True, use_2d_head=True)
    model = build_model(cfg, "freeze", 8, 16, args.checkpoint, device)

    ds = build_dataset_base("freihand", cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    err_old_l, err_pnp_l = [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_batch = {
                'img': batch['img'],
                'mano_params': {'betas': batch['betas']},
                'has_mano_params': {'betas': batch['has_betas']},
            }
            out = model.forward_step(model_batch, train=True)
            pred_keypoints_3d = out['pred_keypoints_3d']
            pred_cam_t = out['pred_cam_t']
            root = pred_keypoints_3d[:, 0:1, :]
            kp3d_rel = pred_keypoints_3d - root

            gt_root = batch['gt_keypoints_3d_abs'][:, 0, :]

            # (1) 모델이 직접 예측한 translation
            T_old = pred_cam_t

            # (2) PnP: 모델이 예측한 2D + 실제 K로 closed-form translation solve
            bsz = kp3d_rel.shape[0]
            box_center = batch['box_center'].view(bsz, 1, 2)
            box_size = batch['box_size'].view(bsz, 1, 1)
            pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
            pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
            T_pnp = solve_translation(kp3d_rel, pred_kp2d_orig, batch['fx'], batch['fy'], batch['px'], batch['py'])

            err_old_l.append((T_old - gt_root).cpu().numpy())
            err_pnp_l.append((T_pnp - gt_root).cpu().numpy())
            n += bsz
            print(f"{n}/{len(ds)} processed", flush=True)

    err_old = np.concatenate(err_old_l, axis=0)
    err_pnp = np.concatenate(err_pnp_l, axis=0)

    result = {
        "dataset": "freihand", "N": n,
        "model_predicted": axis_stats(err_old),
        "pnp": axis_stats(err_pnp),
    }
    print("RESULT:", json.dumps(result, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
