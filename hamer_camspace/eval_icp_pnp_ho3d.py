"""
HO3D eval에 대해 AGILE 방식(ICP -> scale, PnP -> translation)을 우리 HaMeR(exp5)로 시도.

- beta_cond: HO3D eval엔 GT beta가 없어서, 0벡터 대신 mean hand(init_betas,
  mano_mean_params.npz의 shape 필드)를 명시적으로 넣는다.
- R_h(global_orient), theta(hand_pose)는 모델이 예측한 그대로 고정 (MANO forward에
  이미 반영되어 pred_vertices/pred_keypoints_3d로 나옴).
- ICP: depth map(HO3D 공식 decode 공식)을 unproject해서 만든 point cloud에,
  모델이 예측한 root-relative MANO vertices를 trimmed-ICP로 정합 -> scale s_h만 복원
  (R은 이미 모델 예측에 반영돼 있으므로 추가로 안 풂, closed-form).
  hand mask가 없어서(HO3D eval엔 segmentation 미제공) handBoundingBox로 크롭 +
  GT root 깊이 기준 z-band 필터를 "poor man's mask"로 사용 (완전한 대체 아님, 근사).
- PnP: 이미 구현한 solve_translation()으로 translation 복원 (GT-2D / pred-2D 둘 다).
- 최종 CS-MPJPE를 old/GT2D/pred2D/root=GTwrist 대비 비교.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    DATASETS, build_dataset as build_dataset_base, build_cfg, build_model, solve_translation,
)

DEPTH_SCALE = 0.00012498664727900177
HO3D_ROOT = DATASETS["ho3d"]["img_dir"]


def read_depth_img(depth_path):
    img = cv2.imread(depth_path)  # BGR
    dpt = img[:, :, 2].astype(np.float32) + img[:, :, 1].astype(np.float32) * 256.0
    return dpt * DEPTH_SCALE  # meters


class HO3DWrapperWithDepth(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mean_betas):
        self.base = base_dataset
        self.mean_betas = mean_betas.astype(np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # beta_cond: 0 대신 mean hand로 덮어씀 (has_betas=1로 강제)
        item = dict(item)
        item['betas'] = torch.from_numpy(self.mean_betas)
        item['has_betas'] = torch.tensor(1.0)
        return item


def unproject_depth_patch(depth_path, box_center, box_size, fx, fy, px, py, root_depth, z_margin=0.15):
    """handBoundingBox 영역을 크롭하고, GT root 깊이 기준 z-band로 대략적인 hand-only
    point cloud를 만든다 (진짜 mask가 없어서 쓰는 근사)."""
    dpt = read_depth_img(depth_path)
    H, W = dpt.shape
    x0 = int(max(0, box_center[0] - box_size / 2))
    x1 = int(min(W, box_center[0] + box_size / 2))
    y0 = int(max(0, box_center[1] - box_size / 2))
    y1 = int(min(H, box_center[1] + box_size / 2))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = dpt[y0:y1, x0:x1]
    us, vs = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    valid = (patch > 1e-4) & (np.abs(patch - root_depth) < z_margin)
    if valid.sum() < 20:
        return None
    Z = patch[valid]
    U = us[valid].astype(np.float32)
    V = vs[valid].astype(np.float32)
    X = (U - px) * Z / fx
    Y = (V - py) * Z / fy
    return np.stack([X, Y, Z], axis=1).astype(np.float32)  # (N,3)


def trimmed_icp_scale(mesh_pts, cloud_pts, iters=8, trim_ratio=0.7):
    """R은 이미 mesh_pts에 반영돼 있다고 보고, scale s + translation t만 closed-form으로 반복 추정."""
    device = mesh_pts.device
    s = torch.tensor(1.0, device=device)
    t = torch.zeros(3, device=device)
    M = mesh_pts.shape[0]
    k = max(20, int(trim_ratio * M))
    for _ in range(iters):
        transformed = s * mesh_pts + t
        d = torch.cdist(transformed.unsqueeze(0), cloud_pts.unsqueeze(0)).squeeze(0)  # (M,N)
        nn_dist, nn_idx = d.min(dim=1)
        matched = cloud_pts[nn_idx]
        keep = torch.topk(-nn_dist, k).indices
        X = mesh_pts[keep]
        Y = matched[keep]
        Xc = X - X.mean(0)
        Yc = Y - Y.mean(0)
        denom = (Xc * Xc).sum()
        if denom < 1e-8:
            break
        s = (Xc * Yc).sum() / denom
        t = Y.mean(0) - s * X.mean(0)
    return s.item(), t.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=300, help="depth I/O가 느려서 서브셋으로 테스트")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_dir, use_beta_adapter=True, use_2d_head=True)
    model = build_model(cfg, "freeze", 8, 16, args.checkpoint, device)

    mean_betas = np.load(os.path.join(cfg.MANO.DATA_DIR, "mano_mean_params.npz"))['shape']
    print("mean_betas:", mean_betas)

    base_ds = build_dataset_base("ho3d", cfg)
    if args.n_samples > 0:
        idxs = list(range(min(args.n_samples, len(base_ds))))
        base_ds = torch.utils.data.Subset(base_ds, idxs)
    ds = HO3DWrapperWithDepth(base_ds, mean_betas)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # imgname 접근용 (Subset -> HamerEvalWrapper 체인이라 raw npz에서 직접 imgname 인덱싱)
    raw_npz = np.load(DATASETS["ho3d"]["npz"], allow_pickle=True)
    imgnames = raw_npz["imgname"]

    pred_old_l, pred_gt2d_l, pred_pred2d_l, pred_rootgt_l, pred_icp_l, gt_abs_l = [], [], [], [], [], []
    n_icp_ok = 0
    global_idx = 0

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

            pred_old = (kp3d_rel + pred_cam_t.unsqueeze(1)).cpu().numpy()
            T_gt2d = solve_translation(kp3d_rel, batch_gpu['orig_kp2d'], batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
            pred_gt2d = (kp3d_rel + T_gt2d.unsqueeze(1)).cpu().numpy()
            T_rootgt = batch_gpu['gt_keypoints_3d_abs'][:, 0, :]
            pred_rootgt = (kp3d_rel + T_rootgt.unsqueeze(1)).cpu().numpy()

            gt_abs = batch['gt_keypoints_3d_abs'].numpy()

            for b in range(bsz):
                gidx = global_idx + b
                seq = imgnames[gidx].split('/')[1]
                frame_id = os.path.splitext(os.path.basename(imgnames[gidx]))[0]
                depth_path = os.path.join(HO3D_ROOT, 'evaluation', seq, 'depth', frame_id + '.png')

                bc = batch['box_center'][b].numpy()
                bs = float(batch['box_size'][b])
                fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]
                root_depth = gt_abs[b, 0, 2]

                cloud = None
                if os.path.exists(depth_path):
                    cloud = unproject_depth_patch(depth_path, bc, bs, fx_, fy_, px_, py_, root_depth)

                if cloud is not None and len(cloud) >= 20:
                    cloud_t = torch.from_numpy(cloud).to(device)
                    s_h, t_tmp = trimmed_icp_scale(verts_rel[b], cloud_t)
                    kp3d_scaled = kp3d_rel[b] * s_h
                    T_icp = solve_translation(kp3d_scaled.unsqueeze(0), batch_gpu['orig_kp2d'][b:b+1],
                                               batch_gpu['fx'][b:b+1], batch_gpu['fy'][b:b+1],
                                               batch_gpu['px'][b:b+1], batch_gpu['py'][b:b+1])
                    pred_icp = (kp3d_scaled + T_icp[0]).cpu().numpy()
                    n_icp_ok += 1
                else:
                    pred_icp = pred_gt2d[b]  # fallback: depth/point cloud 실패 시 PnP-only 값 사용

                pred_icp_l.append(pred_icp)

            if 'pred_kp2d_mlp_head' in out:
                box_center = batch_gpu['box_center'].view(bsz, 1, 2)
                box_size = batch_gpu['box_size'].view(bsz, 1, 1)
                pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
                pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
                T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
                pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1)).cpu().numpy()
                pred_pred2d_l.append(pred_pred2d)

            pred_old_l.append(pred_old)
            pred_gt2d_l.append(pred_gt2d)
            pred_rootgt_l.append(pred_rootgt)
            gt_abs_l.append(gt_abs)
            global_idx += bsz
            print(f"{global_idx}/{len(ds)} processed, icp_ok={n_icp_ok}", flush=True)

    def cs_mpjpe(preds, gts):
        d = np.linalg.norm(np.stack(preds) - np.concatenate(gts, axis=0), axis=2) * 1000.0
        return float(d.mean()), float(np.median(d))

    gt_abs_all = gt_abs_l
    result = {}
    for name, lst in [("old", pred_old_l), ("gt2d", pred_gt2d_l), ("rootgt", pred_rootgt_l), ("pred2d", pred_pred2d_l)]:
        arr = np.concatenate(lst, axis=0)
        gts = np.concatenate(gt_abs_all, axis=0)
        d = np.linalg.norm(arr - gts, axis=2) * 1000.0
        result[name] = (float(d.mean()), float(d.median()) if hasattr(d, 'median') else float(np.median(d)))

    icp_arr = np.stack(pred_icp_l, axis=0)
    gts_flat = np.concatenate(gt_abs_all, axis=0)
    d_icp = np.linalg.norm(icp_arr - gts_flat, axis=2) * 1000.0
    result['icp_pnp'] = (float(d_icp.mean()), float(np.median(d_icp)))
    result['n_icp_ok'] = n_icp_ok
    result['n_total'] = len(ds)

    print("RESULT:", result)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
