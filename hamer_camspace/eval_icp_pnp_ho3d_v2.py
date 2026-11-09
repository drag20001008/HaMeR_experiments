"""
HO3D eval: scale은 모델 예측 그대로 고정하고(ICP로 안 건드림), translation 복원 방법만
3가지로 나눠서 비교.

- beta_cond: 0 대신 mean hand(init_betas)를 명시적으로 넣음.
- point cloud: SAM3로 미리 뽑아둔 진짜 hand mask(evaluation_hand_masks/<seq>/<frame>.npy)로
  depth를 필터링 (이전 버전의 z-band 근사 대신).
- (1) icp_only: translation-only trimmed ICP (depth point cloud에 mesh를 이동만 시켜 정합)
- (2) pnp_only: 기존 closed-form PnP (GT-2D / pred-2D + K)
- (3) icp_then_pnp: icp_only 결과를 초기값으로 삼아, "진짜" 비선형 reprojection loss를
  Adam으로 짧게 refine (closed-form PnP와 수학적으로 다른 결과가 나오게 하려는 의도).
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')  # "received 0 items of ancdata" fd 고갈 방지
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    DATASETS, build_dataset as build_dataset_base, build_cfg, build_model, solve_translation,
)

DEPTH_SCALE = 0.00012498664727900177
HO3D_ROOT = DATASETS["ho3d"]["img_dir"]
MASK_ROOT = os.path.join(HO3D_ROOT, "evaluation_hand_masks")


def read_depth_img(depth_path):
    img = cv2.imread(depth_path)  # BGR
    dpt = img[:, :, 2].astype(np.float32) + img[:, :, 1].astype(np.float32) * 256.0
    return dpt * DEPTH_SCALE  # meters


def unproject_with_mask(depth_path, mask_path, fx, fy, px, py):
    if not (os.path.exists(depth_path) and os.path.exists(mask_path)):
        return None
    dpt = read_depth_img(depth_path)
    mask = np.load(mask_path)
    if mask.shape != dpt.shape:
        return None
    valid = mask & (dpt > 1e-4)
    if valid.sum() < 20:
        return None
    vs, us = np.nonzero(valid)
    Z = dpt[valid]
    X = (us.astype(np.float32) - px) * Z / fx
    Y = (vs.astype(np.float32) - py) * Z / fy
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


class HO3DWrapperWithDepth(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mean_betas):
        self.base = base_dataset
        self.mean_betas = mean_betas.astype(np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        item['betas'] = torch.from_numpy(self.mean_betas)
        item['has_betas'] = torch.tensor(1.0)
        return item


def trimmed_icp_translation_only(mesh_pts, cloud_pts, iters=8, trim_ratio=0.7):
    """scale은 안 건드리고 translation만 iterative closest point로 복원."""
    t = torch.zeros(3, device=mesh_pts.device)
    M = mesh_pts.shape[0]
    k = max(20, int(trim_ratio * M))
    for _ in range(iters):
        transformed = mesh_pts + t
        d = torch.cdist(transformed.unsqueeze(0), cloud_pts.unsqueeze(0)).squeeze(0)
        nn_dist, nn_idx = d.min(dim=1)
        matched = cloud_pts[nn_idx]
        keep = torch.topk(-nn_dist, k).indices
        t = (matched[keep] - mesh_pts[keep]).mean(0)
    return t


def refine_translation_nonlinear(kp3d_rel, kp2d, fx, fy, px, py, T_init, iters=60, lr=2e-3):
    """closed-form 선형화 없이, 진짜 reprojection loss를 Adam으로 직접 최소화 (ICP 결과에서 출발).
    바깥 루프가 torch.no_grad() 안이라 여기서 로컬로 다시 켜줘야 함."""
    with torch.enable_grad():
        T = T_init.clone().detach().requires_grad_(True)
        kp3d_rel = kp3d_rel.detach()
        opt = torch.optim.Adam([T], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            P = kp3d_rel + T
            u = fx * P[:, 0] / P[:, 2] + px
            v = fy * P[:, 1] / P[:, 2] + py
            loss = ((u - kp2d[:, 0]) ** 2 + (v - kp2d[:, 1]) ** 2).mean()
            loss.backward()
            opt.step()
    return T.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="-1이면 끝까지")
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
    end_idx = len(base_ds) if args.end_idx < 0 else min(args.end_idx, len(base_ds))
    idxs = list(range(args.start_idx, end_idx))
    base_ds = torch.utils.data.Subset(base_ds, idxs)
    print(f"shard: [{args.start_idx}, {end_idx}) -> {len(idxs)} samples")
    ds = HO3DWrapperWithDepth(base_ds, mean_betas)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    raw_npz = np.load(DATASETS["ho3d"]["npz"], allow_pickle=True)
    imgnames = raw_npz["imgname"]

    pred_pnp_gt2d_l, pred_pnp_pred2d_l, pred_icp_only_l, pred_icp_then_pnp_l, gt_abs_l = [], [], [], [], []
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

            # (2) PnP only -- scale 고정(모델 예측 그대로), translation은 2D+K
            T_gt2d = solve_translation(kp3d_rel, batch_gpu['orig_kp2d'], batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
            pred_gt2d = (kp3d_rel + T_gt2d.unsqueeze(1)).cpu().numpy()
            pred_pnp_gt2d_l.append(pred_gt2d)

            box_center = batch_gpu['box_center'].view(bsz, 1, 2)
            box_size = batch_gpu['box_size'].view(bsz, 1, 1)
            pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
            pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
            T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
            pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1)).cpu().numpy()
            pred_pnp_pred2d_l.append(pred_pred2d)

            gt_abs = batch['gt_keypoints_3d_abs'].numpy()
            gt_abs_l.append(gt_abs)

            for b in range(bsz):
                gidx = global_idx + b
                seq = imgnames[gidx].split('/')[1]
                frame_id = os.path.splitext(os.path.basename(imgnames[gidx]))[0]
                depth_path = os.path.join(HO3D_ROOT, 'evaluation', seq, 'depth', frame_id + '.png')
                mask_path = os.path.join(MASK_ROOT, seq, frame_id + '.npy')
                fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]

                cloud = unproject_with_mask(depth_path, mask_path, fx_, fy_, px_, py_)

                if cloud is not None and len(cloud) >= 20:
                    cloud_t = torch.from_numpy(cloud).to(device)
                    # (1) ICP only: scale 고정(=1), translation만 depth로 복원
                    t_icp = trimmed_icp_translation_only(verts_rel[b], cloud_t)
                    pred_icp_only = (kp3d_rel[b] + t_icp).cpu().numpy()

                    # (3) ICP -> PnP: ICP 결과를 초기값으로, 모델이 실제 예측한 2D(pred_kp2d_mlp_head)로
                    # nonlinear reprojection refine (GT 2D는 실전에서 못 쓰니 여기서 쓰면 안 됨)
                    T_refined = refine_translation_nonlinear(
                        kp3d_rel[b], pred_kp2d_orig[b],
                        batch_gpu['fx'][b], batch_gpu['fy'][b], batch_gpu['px'][b], batch_gpu['py'][b],
                        T_init=t_icp)
                    pred_icp_then_pnp = (kp3d_rel[b] + T_refined).cpu().numpy()
                    n_icp_ok += 1
                else:
                    pred_icp_only = pred_pred2d[b]
                    pred_icp_then_pnp = pred_pred2d[b]

                pred_icp_only_l.append(pred_icp_only)
                pred_icp_then_pnp_l.append(pred_icp_then_pnp)

            global_idx += bsz
            print(f"{global_idx}/{len(ds)} processed, icp_ok={n_icp_ok}", flush=True)

    def compute(name, arr_list, is_flat_list):
        gts = np.concatenate(gt_abs_l, axis=0)
        arr = np.stack(arr_list, axis=0) if is_flat_list else np.concatenate(arr_list, axis=0)
        d = np.linalg.norm(arr - gts, axis=2) * 1000.0
        return name, (float(d.mean()), float(np.median(d)))

    result = {}
    for name, val in [
        compute("pnp_only_gt2d", pred_pnp_gt2d_l, False),
        compute("pnp_only_pred2d", pred_pnp_pred2d_l, False) if pred_pnp_pred2d_l else ("pnp_only_pred2d", None),
        compute("icp_only", pred_icp_only_l, True),
        compute("icp_then_pnp", pred_icp_then_pnp_l, True),
    ]:
        result[name] = val
    result['n_icp_ok'] = n_icp_ok
    result['n_total'] = len(ds)

    print("RESULT:", result)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
