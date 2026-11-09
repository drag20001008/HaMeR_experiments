"""
eval_global_positioning.py의 로직을 재사용해서, 몇 개 샘플의
- 3D joint (GT / old weak-persp / GT2D-oracle / pred2D)
- 2D keypoint (GT / 모델이 실제 예측한 pred_kp2d_mlp_head), 둘 다 crop-normalized 좌표
를 JSON으로 덤프. 3D+2D 동시 시각화용.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    DATASETS, build_dataset, build_cfg, build_model, solve_translation,
)


def dump(model, dataset_name, model_cfg, device, n_samples, use_2d_head):
    dataset = build_dataset(dataset_name, model_cfg)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model_batch = {
        'img': batch['img'],
        'mano_params': {'betas': batch['betas']},
        'has_mano_params': {'betas': batch['has_betas']},
    }
    with torch.no_grad():
        out = model.forward_step(model_batch, train=True)
    pred_keypoints_3d = out['pred_keypoints_3d']
    pred_cam_t = out['pred_cam_t']
    root = pred_keypoints_3d[:, 0:1, :]
    kp3d_rel = pred_keypoints_3d - root

    pred_old = (kp3d_rel + pred_cam_t.unsqueeze(1)).cpu().numpy()

    T_gt2d = solve_translation(kp3d_rel, batch['orig_kp2d'], batch['fx'], batch['fy'], batch['px'], batch['py'])
    pred_gt2d = (kp3d_rel + T_gt2d.unsqueeze(1)).cpu().numpy()

    # 2D는 crop-normalized 좌표(pixel_in_crop/224 - 0.5, 대략 [-0.5,0.5])로 통일해서
    # 해상도가 서로 다른 데이터셋끼리도 같은 프레임으로 비교 가능하게 만든다.
    B = kp3d_rel.shape[0]
    box_center = batch['box_center'].view(B, 1, 2)
    box_size = batch['box_size'].view(B, 1, 1)
    gt_kp2d_norm = (batch['orig_kp2d'] - box_center) / box_size

    result = {
        'gt': batch['gt_keypoints_3d_abs'].cpu().numpy().tolist(),
        'old': pred_old.tolist(),
        'gt2d': pred_gt2d.tolist(),
        'gt_kp2d': gt_kp2d_norm.cpu().numpy().tolist(),
    }

    if use_2d_head and 'pred_kp2d_mlp_head' in out:
        pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(B, 21, 2)
        pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
        T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch['fx'], batch['fy'], batch['px'], batch['py'])
        pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1)).cpu().numpy()
        result['pred2d'] = pred_pred2d.tolist()
        result['pred_kp2d'] = pred_kp2d_norm.cpu().numpy().tolist()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_dir, use_beta_adapter=True, use_2d_head=True)
    model = build_model(cfg, "freeze", 8, 16, args.checkpoint, device)

    all_results = {}
    for name in ["freihand", "dexycb"]:
        print(f"dumping {name}...")
        all_results[name] = dump(model, name, cfg, device, args.n_samples, use_2d_head=True)

    with open(args.out_json, "w") as f:
        json.dump(all_results, f)
    print(f"saved to {args.out_json}")


if __name__ == "__main__":
    main()
