"""
HO3D 몇 개 샘플에 대해: PnP(pred-2D)로 배치한 MANO mesh를 PyTorch3D로 depth map/silhouette
렌더링해서, 실측 depth(SAM3 mask 영역 안)/실측 mask(SAM3)랑 직접 비교.
CS-MPJPE(관절 21개)가 아니라 mesh 표면 전체가 실측 depth랑 얼마나 맞는지 보는 독립 검증.
"""
import argparse
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
)

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer

HO3D_ROOT = DATASETS["ho3d"]["img_dir"]
MASK_ROOT = os.path.join(HO3D_ROOT, "evaluation_hand_masks")
DEPTH_SCALE = 0.00012498664727900177
H, W = 480, 640


def read_depth_ho3d(path):
    arr = np.array(Image.open(path).convert("RGB"))  # R,G,B order
    dpt = arr[:, :, 0].astype(np.float32) + arr[:, :, 1].astype(np.float32) * 256.0
    return dpt * DEPTH_SCALE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_dir, use_beta_adapter=True, use_2d_head=True)
    model = build_model(cfg, "freeze", 8, 16, args.checkpoint, device)

    faces = torch.as_tensor(model.mano.faces.astype(np.int64), device=device)  # (F,3)

    ds = build_dataset_base("ho3d", cfg)
    ds = torch.utils.data.Subset(ds, list(range(args.n_samples)))
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=False, num_workers=2)
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

    bsz = args.n_samples
    box_center = batch_gpu['box_center'].view(bsz, 1, 2)
    box_size = batch_gpu['box_size'].view(bsz, 1, 1)
    pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(bsz, 21, 2)
    pred_kp2d_orig = box_center + pred_kp2d_norm * box_size
    T = solve_translation(kp3d_rel, pred_kp2d_orig, batch_gpu['fx'], batch_gpu['fy'], batch_gpu['px'], batch_gpu['py'])
    final_verts = verts_rel + T.unsqueeze(1)  # (B,778,3), 실제 카메라 좌표계

    results = []
    with torch.no_grad():
        for b in range(args.n_samples):
            seq = imgnames[b].split('/')[1]
            frame_id = os.path.splitext(os.path.basename(imgnames[b]))[0]
            depth_path = os.path.join(HO3D_ROOT, 'evaluation', seq, 'depth', frame_id + '.png')
            mask_path = os.path.join(MASK_ROOT, seq, frame_id + '.npy')
            fx_, fy_, px_, py_ = [float(batch[k][b]) for k in ['fx', 'fy', 'px', 'py']]

            if not (os.path.exists(depth_path) and os.path.exists(mask_path)):
                continue
            real_depth = read_depth_ho3d(depth_path)
            real_mask = np.load(mask_path)

            # OpenCV -> PyTorch3D 컨벤션: X,Y 부호 반전
            verts_pt3d = final_verts[b].clone()
            verts_pt3d[:, :2] *= -1

            mesh = Meshes(verts=[verts_pt3d], faces=[faces])
            cameras = PerspectiveCameras(
                focal_length=((fx_, fy_),),
                principal_point=((px_, py_),),
                image_size=((H, W),),
                in_ndc=False,
                device=device,
            )
            raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(mesh)
            rendered_depth = fragments.zbuf[0, ..., 0].cpu().numpy()  # (H,W), 배경 -1
            rendered_mask = rendered_depth > 0

            # depth 오차: 렌더링된 mesh가 덮은 픽셀 AND 실측 mask 픽셀 둘 다인 곳에서만 비교
            overlap = rendered_mask & real_mask
            n_overlap = int(overlap.sum())
            if n_overlap > 20:
                depth_err_mm = float(np.abs(rendered_depth[overlap] - real_depth[overlap]).mean() * 1000.0)
            else:
                depth_err_mm = None

            inter = int((rendered_mask & real_mask).sum())
            union = int((rendered_mask | real_mask).sum())
            iou = float(inter / union) if union > 0 else None

            print(f"sample {b}: depth_err={depth_err_mm} mm, iou={iou}, n_overlap={n_overlap}", flush=True)
            results.append({"idx": b, "depth_err_mm": depth_err_mm, "iou": iou, "n_overlap": n_overlap})

    valid_depth = [r["depth_err_mm"] for r in results if r["depth_err_mm"] is not None]
    valid_iou = [r["iou"] for r in results if r["iou"] is not None]
    summary = {
        "per_sample": results,
        "mean_depth_err_mm": float(np.mean(valid_depth)) if valid_depth else None,
        "mean_iou": float(np.mean(valid_iou)) if valid_iou else None,
    }
    print("SUMMARY:", summary)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
