"""
DexYCB: wrist(root) translation을 세 가지 방식(Model-Predicted / PnP / ICP->PnP)으로 복원했을 때,
GT wrist 절대좌표와의 오차를 X/Y/Z 축별로 분리해서 계산 (샤딩 지원, full set용).
"""
import argparse
import json
import os

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    DATASETS, build_dataset as build_dataset_base, build_cfg, build_model, solve_translation,
    unmirror_left_hand,
)
from eval_icp_pnp_ho3d_v2 import trimmed_icp_translation_only, refine_translation_nonlinear
from eval_icp_pnp_dexycb import DEXYCB_ROOT, unproject_with_seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_dir, use_beta_adapter=True, use_2d_head=True)
    model = build_model(cfg, "freeze", 8, 16, args.checkpoint, device)

    ds = build_dataset_base("dexycb", cfg)
    end_idx = len(ds) if args.end_idx < 0 else min(args.end_idx, len(ds))
    idxs = list(range(args.start_idx, end_idx))
    ds = torch.utils.data.Subset(ds, idxs)
    print(f"shard: [{args.start_idx}, {end_idx}) -> {len(idxs)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    raw_npz = np.load(DATASETS["dexycb"]["npz"], allow_pickle=True)
    test_indices = np.where(raw_npz["official_split"] == "test")[0]
    imgnames = raw_npz["imgname"]

    err_old_l, err_pnp_l, err_icppnp_l = [], [], []
    n_icp_ok = 0
    global_idx = args.start_idx

    with torch.no_grad():
        for batch in loader:
            bsz = batch['img'].shape[0]
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_batch = {
                'img': batch_gpu['img'],
                'mano_params': {'betas': batch_gpu['betas']},
                'has_mano_params': {'betas': batch_gpu['has_betas']},
            }
            out = model.forward_step(model_batch, train=True)
            pred_keypoints_3d = out['pred_keypoints_3d']
            pred_vertices = out['pred_vertices']
            pred_cam_t = out['pred_cam_t']
            root = pred_keypoints_3d[:, 0:1, :]
            kp3d_rel = pred_keypoints_3d - root
            verts_rel = pred_vertices - root

            right = batch_gpu['right']
            kp3d_rel = unmirror_left_hand(kp3d_rel, right)
            verts_rel = unmirror_left_hand(verts_rel, right)
            gt_abs_t = unmirror_left_hand(batch_gpu['gt_keypoints_3d_abs'], right)
            T_old = unmirror_left_hand(pred_cam_t, right)

            gt_root = gt_abs_t[:, 0, :]

            box_center = batch_gpu['box_center'].view(bsz, 1, 2)
            box_size = batch_gpu['box_size'].view(bsz, 1, 1)
            pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
            pred_kp2d_norm = unmirror_left_hand(pred_kp2d_norm, right)
            pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
            T_pnp = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])

            err_old_l.append((T_old - gt_root).cpu().numpy())
            err_pnp_l.append((T_pnp - gt_root).cpu().numpy())

            for b in range(bsz):
                gidx = global_idx + b
                orig_row = test_indices[gidx]
                rel = imgnames[orig_row]
                folder, fname = os.path.split(rel)
                frame_num = fname.replace("color_", "").replace(".jpg", "")
                depth_path = os.path.join(DEXYCB_ROOT, folder, f"aligned_depth_to_color_{frame_num}.png")
                labels_path = os.path.join(DEXYCB_ROOT, folder, f"labels_{frame_num}.npz")
                fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]

                cloud = unproject_with_seg(depth_path, labels_path, fx_, fy_, px_, py_)
                if cloud is not None and len(cloud) >= 20:
                    cloud_t = torch.from_numpy(cloud).to(device)
                    t_icp = trimmed_icp_translation_only(verts_rel[b], cloud_t)
                    T_refined = refine_translation_nonlinear(
                        kp3d_rel[b], pred_kp2d_orig[b],
                        batch_gpu['fx'][b], batch_gpu['fy'][b], batch_gpu['px'][b], batch_gpu['py'][b],
                        T_init=t_icp)
                    n_icp_ok += 1
                else:
                    T_refined = T_pnp[b]

                err_icppnp_l.append((T_refined - gt_root[b]).cpu().numpy())

            global_idx += bsz
            print(f"{global_idx}/{end_idx} processed, icp_ok={n_icp_ok}", flush=True)

    err_old = np.concatenate(err_old_l, axis=0)
    err_pnp = np.concatenate(err_pnp_l, axis=0)
    err_icppnp = np.stack(err_icppnp_l, axis=0)

    result = {
        "err_old": err_old.tolist(),
        "err_pnp": err_pnp.tolist(),
        "err_icppnp": err_icppnp.tolist(),
        "n_icp_ok": n_icp_ok,
        "n_total": len(idxs),
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print(f"saved raw per-sample errors ({len(idxs)} samples) to {args.out_json}")


if __name__ == "__main__":
    main()
