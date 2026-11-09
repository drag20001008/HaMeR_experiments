"""
AGILE 논문 그대로의 절차: (1) ICP로 rotation은 고정한 채 scale(s_h)과 임시 translation을 depth에
맞춰 복원, (2) scale이 보정된 3D joint + 모델이 예측한 2D(J_2D, 원본 HaMeR의 native
pred_keypoints_2d) + 실제 카메라 K로 PnP를 풀어 최종 translation(T_h)을 복원.
DexYCB 전체(샤딩)에 대해 wrist(root) translation 오차를 X/Y/Z 축별로 계산.
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
    DATASETS, build_dataset as build_dataset_base, build_cfg, solve_translation,
    unmirror_left_hand,
)
from eval_icp_pnp_dexycb import DEXYCB_ROOT, unproject_with_seg
from hamer.models.hamer import HAMER


def build_model_original(cfg, checkpoint_path, device):
    model = HAMER(cfg, init_renderer=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded ORIGINAL checkpoint {checkpoint_path}")
    print(f"missing={missing}")
    print(f"unexpected={[k for k in unexpected if not k.startswith('discriminator.')]}")
    model = model.to(device)
    model.eval()
    return model


def trimmed_icp_scale_translation(mesh_pts, cloud_pts, iters=8, trim_ratio=0.7):
    """AGILE: rotation(R_h) 고정, scale(s_h) + 임시 translation만 ICP로 복원.
    각 iteration마다: (1) 현재 (s,t)로 변환한 mesh와 depth cloud 최근접 대응 찾기,
    (2) 대응쌍으로 스칼라 scale+translation을 닫힌 형태로 재추정."""
    s = torch.ones((), device=mesh_pts.device)
    t = torch.zeros(3, device=mesh_pts.device)
    M = mesh_pts.shape[0]
    k = max(20, int(trim_ratio * M))
    for _ in range(iters):
        transformed = s * mesh_pts + t
        d = torch.cdist(transformed.unsqueeze(0), cloud_pts.unsqueeze(0)).squeeze(0)
        nn_dist, nn_idx = d.min(dim=1)
        matched = cloud_pts[nn_idx]
        keep = torch.topk(-nn_dist, k).indices
        p = mesh_pts[keep]
        q = matched[keep]
        p_mean = p.mean(0)
        q_mean = q.mean(0)
        p_c = p - p_mean
        q_c = q - q_mean
        denom = (p_c * p_c).sum().clamp_min(1e-9)
        s = (p_c * q_c).sum() / denom
        t = q_mean - s * p_mean
    return s.detach(), t.detach()


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
    cfg = build_cfg(args.config_dir, use_beta_adapter=False, use_2d_head=False)
    model = build_model_original(cfg, args.checkpoint, device)

    ds = build_dataset_base("dexycb", cfg)
    end_idx = len(ds) if args.end_idx < 0 else min(args.end_idx, len(ds))
    idxs = list(range(args.start_idx, end_idx))
    ds = torch.utils.data.Subset(ds, idxs)
    print(f"shard: [{args.start_idx}, {end_idx}) -> {len(idxs)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    raw_npz = np.load(DATASETS["dexycb"]["npz"], allow_pickle=True)
    test_indices = np.where(raw_npz["official_split"] == "test")[0]
    imgnames = raw_npz["imgname"]

    err_agile_l = []
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
            root = pred_keypoints_3d[:, 0:1, :]
            kp3d_rel = pred_keypoints_3d - root
            verts_rel = pred_vertices - root

            right = batch_gpu['right']
            kp3d_rel = unmirror_left_hand(kp3d_rel, right)
            verts_rel = unmirror_left_hand(verts_rel, right)
            gt_abs_t = unmirror_left_hand(batch_gpu['gt_keypoints_3d_abs'], right)
            gt_root = gt_abs_t[:, 0, :]

            box_center = batch_gpu['box_center'].view(bsz, 1, 2)
            box_size = batch_gpu['box_size'].view(bsz, 1, 1)
            pred_kp2d_norm = out['pred_keypoints_2d'].view(bsz, 21, 2)
            pred_kp2d_norm = unmirror_left_hand(pred_kp2d_norm, right)
            pred_kp2d_orig = box_center + pred_kp2d_norm * box_size

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
                    s_h, _t_tmp = trimmed_icp_scale_translation(verts_rel[b], cloud_t)
                    n_icp_ok += 1
                else:
                    s_h = torch.ones((), device=device)

                # scale이 보정된 3D + 예측 2D + 실제 K로 PnP -> 최종 translation
                kp3d_scaled = kp3d_rel[b] * s_h
                fx_b, fy_b, px_b, py_b = batch_gpu['fx'][b:b+1], batch_gpu['fy'][b:b+1], batch_gpu['px'][b:b+1], batch_gpu['py'][b:b+1]
                T_h = solve_translation(kp3d_scaled.unsqueeze(0), pred_kp2d_orig[b:b+1], fx_b, fy_b, px_b, py_b)[0]

                err_agile_l.append((T_h - gt_root[b]).cpu().numpy())

            global_idx += bsz
            print(f"{global_idx}/{end_idx} processed, icp_ok={n_icp_ok}", flush=True)

    err_agile = np.stack(err_agile_l, axis=0)

    result = {
        "err_agile_icp_scale_pnp": err_agile.tolist(),
        "n_icp_ok": n_icp_ok,
        "n_total": len(idxs),
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print(f"saved raw per-sample errors ({len(idxs)} samples) to {args.out_json}")


if __name__ == "__main__":
    main()
