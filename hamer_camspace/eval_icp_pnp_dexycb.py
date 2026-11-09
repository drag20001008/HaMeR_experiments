"""
DexYCB eval: HO3D 버전(eval_icp_pnp_ho3d_v2.py)과 동일한 3가지 translation 복원 비교
(icp_only / pnp_only / icp_then_pnp)를, DexYCB의 공식 GT segmentation(labels_*.npz의
'seg'==255=hand)과 표준 RealSense depth(16bit, mm)로 수행.

- DexYCB는 GT beta가 원래부터 있음(has_betas=1.0) -> mean-hand 치환 불필요.
- depth: aligned_depth_to_color_XXXXXX.png, uint16 mm -> *0.001로 meter 변환.
- mask: labels_XXXXXX.npz의 'seg' 필드에서 255(hand)인 픽셀만.
"""
import argparse
import json
import os

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')  # "received 0 items of ancdata" fd 고갈 방지
from PIL import Image
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    DATASETS, build_dataset as build_dataset_base, build_cfg, build_model, solve_translation,
    unmirror_left_hand,
)
from eval_icp_pnp_ho3d_v2 import trimmed_icp_translation_only, refine_translation_nonlinear

DEXYCB_ROOT = DATASETS["dexycb"]["img_dir"]
HAND_LABEL = 255


def read_depth_mm(depth_path):
    arr = np.array(Image.open(depth_path))  # uint16, mm
    return arr.astype(np.float32) * 0.001  # meters


def unproject_with_seg(depth_path, labels_path, fx, fy, px, py):
    if not (os.path.exists(depth_path) and os.path.exists(labels_path)):
        return None
    dpt = read_depth_mm(depth_path)
    seg = np.load(labels_path)['seg']
    if seg.shape != dpt.shape:
        return None
    valid = (seg == HAND_LABEL) & (dpt > 1e-4)
    if valid.sum() < 20:
        return None
    vs, us = np.nonzero(valid)
    Z = dpt[valid]
    X = (us.astype(np.float32) - px) * Z / fx
    Y = (vs.astype(np.float32) - py) * Z / fy
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


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

    ds = build_dataset_base("dexycb", cfg)  # Subset(test split) 이미 적용됨, GT beta도 이미 있음
    end_idx = len(ds) if args.end_idx < 0 else min(args.end_idx, len(ds))
    idxs = list(range(args.start_idx, end_idx))
    ds = torch.utils.data.Subset(ds, idxs)
    print(f"shard: [{args.start_idx}, {end_idx}) -> {len(idxs)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    raw_npz = np.load(DATASETS["dexycb"]["npz"], allow_pickle=True)
    test_indices = np.where(raw_npz["official_split"] == "test")[0]
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

            # 왼손(right==0)은 get_example이 flip해서 넣은 거라, 실제 카메라 좌표계
            # (orig_kp2d/depth/K)와 맞추려면 x부호를 되돌려야 함
            right = batch_gpu['right']
            kp3d_rel = unmirror_left_hand(kp3d_rel, right)
            verts_rel = unmirror_left_hand(verts_rel, right)
            gt_abs_t = unmirror_left_hand(batch_gpu['gt_keypoints_3d_abs'], right)

            T_gt2d = solve_translation(kp3d_rel, batch_gpu['orig_kp2d'], batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
            pred_gt2d = (kp3d_rel + T_gt2d.unsqueeze(1)).cpu().numpy()
            pred_pnp_gt2d_l.append(pred_gt2d)

            box_center = batch_gpu['box_center'].view(bsz, 1, 2)
            box_size = batch_gpu['box_size'].view(bsz, 1, 1)
            pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
            pred_kp2d_norm = unmirror_left_hand(pred_kp2d_norm, right)
            pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
            T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
            pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1)).cpu().numpy()
            pred_pnp_pred2d_l.append(pred_pred2d)

            gt_abs = gt_abs_t.cpu().numpy()
            gt_abs_l.append(gt_abs)

            for b in range(bsz):
                gidx = global_idx + b
                orig_row = test_indices[gidx]
                rel = imgnames[orig_row]  # "<subject>/<seq>/<cam>/color_XXXXXX.jpg"
                folder, fname = os.path.split(rel)
                frame_num = fname.replace("color_", "").replace(".jpg", "")
                depth_path = os.path.join(DEXYCB_ROOT, folder, f"aligned_depth_to_color_{frame_num}.png")
                labels_path = os.path.join(DEXYCB_ROOT, folder, f"labels_{frame_num}.npz")
                fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]

                cloud = unproject_with_seg(depth_path, labels_path, fx_, fy_, px_, py_)

                if cloud is not None and len(cloud) >= 20:
                    cloud_t = torch.from_numpy(cloud).to(device)
                    t_icp = trimmed_icp_translation_only(verts_rel[b], cloud_t)
                    pred_icp_only = (kp3d_rel[b] + t_icp).cpu().numpy()

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
