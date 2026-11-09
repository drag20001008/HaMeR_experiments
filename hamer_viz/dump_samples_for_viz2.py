"""
HO3D, DexYCB 각각 30개 샘플에 대해 6가지 3D pose(GT/pred2D+K/root=GTwrist/ICP만/PnP만(GT2D)/
ICP->PnP) + 입력 RGB(base64 JPEG) + depth point cloud(ICP에 쓴 것) + hand mask 오버레이(base64 PNG)를
JSON으로 덤프. 3D+RGB+point cloud+mask 동시 시각화용.
"""
import argparse
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from eval_global_positioning import (
    DATASETS, build_dataset as build_dataset_base, build_cfg, build_model, solve_translation,
    unmirror_left_hand,
)
from eval_icp_pnp_ho3d_v2 import (
    HO3D_ROOT, MASK_ROOT, HO3DWrapperWithDepth, unproject_with_mask,
    trimmed_icp_translation_only, refine_translation_nonlinear, read_depth_img,
)
from eval_icp_pnp_dexycb import DEXYCB_ROOT, unproject_with_seg, HAND_LABEL

MAX_CLOUD_PTS = 1500
IMG_MAX_W = 320


def compute_resize(w, h, max_w):
    if w <= max_w:
        return w, h
    return max_w, int(h * max_w / w)


def img_to_b64(rgb_path, max_w=IMG_MAX_W):
    img = Image.open(rgb_path).convert("RGB")
    w, h = img.size
    tw, th = compute_resize(w, h, max_w)
    if (tw, th) != (w, h):
        img = img.resize((tw, th), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), (tw, th)


def mask_to_overlay_b64(mask_bool, target_size, color=(239, 68, 68)):
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_bool] = [color[0], color[1], color[2], 150]
    img = Image.fromarray(rgba, mode="RGBA").resize(target_size, Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def subsample_cloud(cloud):
    if cloud is None:
        return None
    if len(cloud) > MAX_CLOUD_PTS:
        idx = np.random.choice(len(cloud), MAX_CLOUD_PTS, replace=False)
        cloud = cloud[idx]
    return cloud.tolist()


def diverse_indices(total_len, n_samples):
    # 데이터셋이 시퀀스/subject 순서로 저장돼 있어서 range(n_samples)(앞부분만)로 뽑으면
    # 전부 같은 시퀀스에서만 나옴 -- 전체 구간에 걸쳐 균등 간격으로 뽑아서 다양한 시퀀스/조명/물체가 섞이게 함
    if n_samples >= total_len:
        return list(range(total_len))
    return np.linspace(0, total_len - 1, n_samples).round().astype(int).tolist()


def process_ho3d(model, cfg, device, n_samples):
    mean_betas = np.load(os.path.join(cfg.MANO.DATA_DIR, "mano_mean_params.npz"))['shape']
    base_ds_full = build_dataset_base("ho3d", cfg)
    idxs = diverse_indices(len(base_ds_full), n_samples)
    n_samples = len(idxs)
    base_ds = torch.utils.data.Subset(base_ds_full, idxs)
    ds = HO3DWrapperWithDepth(base_ds, mean_betas)
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    raw_npz = np.load(DATASETS["ho3d"]["npz"], allow_pickle=True)
    imgnames = raw_npz["imgname"]

    model_batch = {
        'img': batch_gpu['img'],
        'mano_params': {'betas': batch_gpu['betas']},
        'has_mano_params': {'betas': batch_gpu['has_betas']},
    }
    with torch.no_grad():
        out = model.forward_step(model_batch, train=True)
    pred_keypoints_3d = out['pred_keypoints_3d']
    pred_vertices = out['pred_vertices']
    root = pred_keypoints_3d[:, 0:1, :]
    kp3d_rel = pred_keypoints_3d - root
    verts_rel = pred_vertices - root

    bsz = n_samples
    box_center = batch_gpu['box_center'].view(bsz, 1, 2)
    box_size = batch_gpu['box_size'].view(bsz, 1, 1)
    pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
    pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
    # PnP는 어디서도 GT 2D를 쓰지 않는다 -- 실제 예측된 2D(pred_kp2d_mlp_head)로만 품
    T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
    pred_pnp = (kp3d_rel + T_pred2d.unsqueeze(1))
    pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1))

    T_rootgt = batch_gpu['gt_keypoints_3d_abs'][:, 0, :]
    pred_rootgt = (kp3d_rel + T_rootgt.unsqueeze(1))

    gt_abs = batch['gt_keypoints_3d_abs'].numpy()

    icp_only_l, icp_then_pnp_l, rgb_l, cloud_l, mask_l = [], [], [], [], []
    with torch.no_grad():
        for b in range(n_samples):
            imgname_b = imgnames[idxs[b]]
            seq = imgname_b.split('/')[1]
            frame_id = os.path.splitext(os.path.basename(imgname_b))[0]
            depth_path = os.path.join(HO3D_ROOT, 'evaluation', seq, 'depth', frame_id + '.png')
            mask_path = os.path.join(MASK_ROOT, seq, frame_id + '.npy')
            rgb_path = os.path.join(HO3D_ROOT, imgname_b)
            fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]

            rgb_b64, target_size = img_to_b64(rgb_path)
            rgb_l.append(rgb_b64)

            mask_b64 = None
            if os.path.exists(mask_path):
                mask_arr = np.load(mask_path)
                mask_b64 = mask_to_overlay_b64(mask_arr, target_size)
            mask_l.append(mask_b64)

            cloud = unproject_with_mask(depth_path, mask_path, fx_, fy_, px_, py_)
            cloud_l.append(subsample_cloud(cloud))

            if cloud is not None and len(cloud) >= 20:
                cloud_t = torch.from_numpy(cloud).to(device)
                t_icp = trimmed_icp_translation_only(verts_rel[b], cloud_t)
                pred_icp = (kp3d_rel[b] + t_icp)
                T_ref = refine_translation_nonlinear(
                    kp3d_rel[b], pred_kp2d_orig[b],
                    batch_gpu['fx'][b], batch_gpu['fy'][b], batch_gpu['px'][b], batch_gpu['py'][b], T_init=t_icp)
                pred_icp_pnp = (kp3d_rel[b] + T_ref)
            else:
                pred_icp = pred_pred2d[b]
                pred_icp_pnp = pred_pred2d[b]

            icp_only_l.append(pred_icp.cpu().numpy())
            icp_then_pnp_l.append(pred_icp_pnp.cpu().numpy())

    return {
        'gt': gt_abs.tolist(),
        'pred2d': pred_pred2d.cpu().numpy().tolist(),
        'rootgt': pred_rootgt.cpu().numpy().tolist(),
        'icp': np.stack(icp_only_l).tolist(),
        'pnp': pred_pnp.cpu().numpy().tolist(),
        'icp_pnp': np.stack(icp_then_pnp_l).tolist(),
        'rgb': rgb_l,
        'mask': mask_l,
        'cloud': cloud_l,
    }


def process_dexycb(model, cfg, device, n_samples):
    ds_full = build_dataset_base("dexycb", cfg)
    idxs = diverse_indices(len(ds_full), n_samples)
    n_samples = len(idxs)
    ds = torch.utils.data.Subset(ds_full, idxs)
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    raw_npz = np.load(DATASETS["dexycb"]["npz"], allow_pickle=True)
    test_indices = np.where(raw_npz["official_split"] == "test")[0]
    imgnames = raw_npz["imgname"]

    model_batch = {
        'img': batch_gpu['img'],
        'mano_params': {'betas': batch_gpu['betas']},
        'has_mano_params': {'betas': batch_gpu['has_betas']},
    }
    with torch.no_grad():
        out = model.forward_step(model_batch, train=True)
    pred_keypoints_3d = out['pred_keypoints_3d']
    pred_vertices = out['pred_vertices']
    root = pred_keypoints_3d[:, 0:1, :]
    kp3d_rel = pred_keypoints_3d - root
    verts_rel = pred_vertices - root

    # 왼손(right==0)은 get_example이 flip해서 넣은 거라, 실제 카메라 좌표계로 되돌림
    right = batch_gpu['right']
    kp3d_rel = unmirror_left_hand(kp3d_rel, right)
    verts_rel = unmirror_left_hand(verts_rel, right)
    gt_abs_t = unmirror_left_hand(batch_gpu['gt_keypoints_3d_abs'], right)

    bsz = n_samples
    box_center = batch_gpu['box_center'].view(bsz, 1, 2)
    box_size = batch_gpu['box_size'].view(bsz, 1, 1)
    pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
    pred_kp2d_norm = unmirror_left_hand(pred_kp2d_norm, right)
    pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
    # PnP는 어디서도 GT 2D를 쓰지 않는다 -- 실제 예측된 2D(pred_kp2d_mlp_head)로만 품
    T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
    pred_pnp = (kp3d_rel + T_pred2d.unsqueeze(1))
    pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1))

    T_rootgt = gt_abs_t[:, 0, :]
    pred_rootgt = (kp3d_rel + T_rootgt.unsqueeze(1))

    gt_abs = gt_abs_t.cpu().numpy()

    icp_only_l, icp_then_pnp_l, rgb_l, cloud_l, mask_l = [], [], [], [], []
    with torch.no_grad():
        for b in range(n_samples):
            orig_row = test_indices[idxs[b]]
            rel = imgnames[orig_row]
            folder, fname = os.path.split(rel)
            frame_num = fname.replace("color_", "").replace(".jpg", "")
            depth_path = os.path.join(DEXYCB_ROOT, folder, f"aligned_depth_to_color_{frame_num}.png")
            labels_path = os.path.join(DEXYCB_ROOT, folder, f"labels_{frame_num}.npz")
            rgb_path = os.path.join(DEXYCB_ROOT, rel)
            fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]

            rgb_b64, target_size = img_to_b64(rgb_path)
            rgb_l.append(rgb_b64)

            mask_b64 = None
            if os.path.exists(labels_path):
                seg = np.load(labels_path)['seg']
                mask_b64 = mask_to_overlay_b64(seg == HAND_LABEL, target_size)
            mask_l.append(mask_b64)

            cloud = unproject_with_seg(depth_path, labels_path, fx_, fy_, px_, py_)
            cloud_l.append(subsample_cloud(cloud))

            if cloud is not None and len(cloud) >= 20:
                cloud_t = torch.from_numpy(cloud).to(device)
                t_icp = trimmed_icp_translation_only(verts_rel[b], cloud_t)
                pred_icp = (kp3d_rel[b] + t_icp)
                T_ref = refine_translation_nonlinear(
                    kp3d_rel[b], pred_kp2d_orig[b],
                    batch_gpu['fx'][b], batch_gpu['fy'][b], batch_gpu['px'][b], batch_gpu['py'][b], T_init=t_icp)
                pred_icp_pnp = (kp3d_rel[b] + T_ref)
            else:
                pred_icp = pred_pred2d[b]
                pred_icp_pnp = pred_pred2d[b]

            icp_only_l.append(pred_icp.cpu().numpy())
            icp_then_pnp_l.append(pred_icp_pnp.cpu().numpy())

    return {
        'gt': gt_abs.tolist(),
        'pred2d': pred_pred2d.cpu().numpy().tolist(),
        'rootgt': pred_rootgt.cpu().numpy().tolist(),
        'icp': np.stack(icp_only_l).tolist(),
        'pnp': pred_pnp.cpu().numpy().tolist(),
        'icp_pnp': np.stack(icp_then_pnp_l).tolist(),
        'rgb': rgb_l,
        'mask': mask_l,
        'cloud': cloud_l,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_dir, use_beta_adapter=True, use_2d_head=True)
    model = build_model(cfg, "freeze", 8, 16, args.checkpoint, device)

    result = {}
    print("processing ho3d...")
    result['ho3d'] = process_ho3d(model, cfg, device, args.n_samples)
    print("processing dexycb...")
    result['dexycb'] = process_dexycb(model, cfg, device, args.n_samples)

    with open(args.out_json, "w") as f:
        json.dump(result, f)
    print("saved to", args.out_json)


if __name__ == "__main__":
    main()
