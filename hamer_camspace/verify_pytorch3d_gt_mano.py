"""
모델 inference 없이, HO3D **train** split의 진짜 GT MANO 파라미터(handPose/handBeta/handTrans)로
mesh를 직접 만들어서 PyTorch3D로 렌더링 -> 실측 depth와 비교.
목적: rasterizer(좌표계 변환, K 매핑, z-buffer)가 그 자체로 맞게 구현됐는지 순수 검증
(모델 예측 정확도와 완전히 분리).

HO3D eval split엔 GT pose/beta가 없어서(공식적으로 숨김), train split만 사용 가능.
좌표계: HO3D 원본은 OpenGL 컨벤션 -> gl_to_cv로 표준 CV 컨벤션 변환 후 카메라 K 적용.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from hamer.models.mano_wrapper import MANO
from smplx.lbs import batch_rodrigues

MANO_MODEL_PATH = "/workspace/drag1008/hamer/_DATA/data/mano"

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer

HO3D_ROOT = "/nas2/lait/1000_Members/jaehoon/HO3D_v3"
MASK_ROOT = os.path.join(HO3D_ROOT, "train_hand_masks")
DEPTH_SCALE = 0.00012498664727900177
H, W = 480, 640

SAMPLES = [
    ("MC1", "0000"),
    ("ABF10", "0000"),
    ("SM2", "0000"),
    ("GPMF10", "0000"),
    ("ShSu10", "0000"),
]


def gl_to_cv(points):
    out = points.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


def read_depth_ho3d(path):
    arr = np.array(Image.open(path).convert("RGB"))
    dpt = arr[:, :, 0].astype(np.float32) + arr[:, :, 1].astype(np.float32) * 256.0
    return dpt * DEPTH_SCALE


def project(xyz, K):
    uv = (K @ xyz.T).T
    return uv[:, :2] / uv[:, 2:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--viz_dir", default=None)
    args = parser.parse_args()
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano = MANO(model_path=MANO_MODEL_PATH, gender="neutral", num_hand_joints=15,
                create_body_pose=False, use_pca=False, flat_hand_mean=True).to(device)
    faces = torch.as_tensor(mano.faces.astype(np.int64), device=device)

    results = []
    for seq, frame_id in SAMPLES:
        meta_path = os.path.join(HO3D_ROOT, "train", seq, "meta", frame_id + ".pkl")
        depth_path = os.path.join(HO3D_ROOT, "train", seq, "depth", frame_id + ".png")
        if not (os.path.exists(meta_path) and os.path.exists(depth_path)):
            print(f"skip {seq}/{frame_id}: missing files")
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        hand_pose_full = np.asarray(meta["handPose"], dtype=np.float32)  # (48,)
        hand_beta = np.asarray(meta["handBeta"], dtype=np.float32)       # (10,)
        hand_trans = np.asarray(meta["handTrans"], dtype=np.float32)    # (3,)
        cam_mat = np.asarray(meta["camMat"], dtype=np.float32)
        fx, fy, px, py = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]

        # HO3D GT handPose는 axis-angle(48=3+45)인데, HaMeR의 MANO(smplx.MANOLayer)는
        # forward에서 항상 pose2rot=False로 lbs를 호출 -- 즉 global_orient/hand_pose를
        # 이미 회전행렬(rotmat)로 기대함. 그래서 axis-angle -> rotmat 변환을 직접 해줘야 함.
        global_orient_aa = torch.from_numpy(hand_pose_full[:3]).float().unsqueeze(0).to(device)  # (1,3)
        hand_pose_aa = torch.from_numpy(hand_pose_full[3:]).float().view(15, 3).to(device)        # (15,3)
        global_orient = batch_rodrigues(global_orient_aa).view(1, 1, 3, 3)
        hand_pose = batch_rodrigues(hand_pose_aa).view(1, 15, 3, 3)
        betas = torch.from_numpy(hand_beta).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas)
        verts_gl = out.vertices[0].cpu().numpy()          # MANO 자체 canonical frame, GL 컨벤션
        verts_gl_abs = verts_gl + hand_trans[None, :]      # GT translation 적용 (여전히 GL)
        verts_cv_abs = gl_to_cv(verts_gl_abs)              # CV 컨벤션으로 변환 (real camMat과 맞는 좌표계)

        # 비교 영역: bbox가 아니라 SAM3로 뽑은 실제 손 segmentation 마스크
        mask_path = os.path.join(MASK_ROOT, seq, frame_id + ".npy")
        if not os.path.exists(mask_path):
            print(f"skip {seq}/{frame_id}: no sam3 mask")
            continue
        sam3_mask = np.load(mask_path)

        real_depth = read_depth_ho3d(depth_path)

        verts_pt3d = torch.from_numpy(verts_cv_abs).float().to(device)
        verts_pt3d[:, :2] *= -1  # OpenCV -> PyTorch3D 컨벤션
        mesh = Meshes(verts=[verts_pt3d], faces=[faces])
        cameras = PerspectiveCameras(
            focal_length=((float(fx), float(fy)),),
            principal_point=((float(px), float(py)),),
            image_size=((H, W),),
            in_ndc=False,
            device=device,
        )
        raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)
        rendered_depth = fragments.zbuf[0, ..., 0].cpu().numpy()
        rendered_mask = rendered_depth > 0

        overlap = rendered_mask & sam3_mask & (real_depth > 1e-4)
        n_overlap = int(overlap.sum())
        depth_err_mm = float(np.abs(rendered_depth[overlap] - real_depth[overlap]).mean() * 1000.0) if n_overlap > 20 else None

        print(f"{seq}/{frame_id}: n_rendered_px={int(rendered_mask.sum())}, n_overlap={n_overlap}, depth_err_mm={depth_err_mm}", flush=True)
        results.append({"seq": seq, "frame": frame_id, "n_rendered_px": int(rendered_mask.sum()),
                         "n_overlap": n_overlap, "depth_err_mm": depth_err_mm})

        if args.viz_dir:
            rgb_path = os.path.join(HO3D_ROOT, "train", seq, "rgb", frame_id + ".jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(HO3D_ROOT, "train", seq, "rgb", frame_id + ".png")
            rgb = np.array(Image.open(rgb_path).convert("RGB")) if os.path.exists(rgb_path) else np.zeros((H, W, 3), dtype=np.uint8)

            # rendered mesh 실루엣(초록) + SAM3 손 마스크 경계(빨강) + 교집합(파랑)을 RGB에 오버레이
            rendered_boundary = rendered_mask & ~ndimage.binary_erosion(rendered_mask, iterations=1)
            sam3_boundary = sam3_mask & ~ndimage.binary_erosion(sam3_mask, iterations=1)
            rgb_overlay = rgb.copy()
            rgb_overlay[rendered_boundary] = [0, 255, 0]
            rgb_overlay[sam3_boundary] = [255, 0, 0]
            rgb_overlay[overlap] = (0.5 * rgb_overlay[overlap] + 0.5 * np.array([0, 120, 255])).astype(np.uint8)

            valid_depth_mask = real_depth > 1e-4
            vmin = real_depth[valid_depth_mask].min() if valid_depth_mask.any() else 0.0
            vmax = real_depth[valid_depth_mask].max() if valid_depth_mask.any() else 1.0

            fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
            axes[0].imshow(rgb_overlay)
            axes[0].set_title(f"{seq}/{frame_id}: rendered(green) / SAM3 mask(red) / overlap(blue)")
            axes[0].axis("off")

            im1 = axes[1].imshow(np.where(valid_depth_mask, real_depth, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
            axes[1].set_title("real depth (m)")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            im2 = axes[2].imshow(np.where(rendered_mask, rendered_depth, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
            axes[2].set_title("rendered GT-mesh depth (m)")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            diff = np.full((H, W), np.nan, dtype=np.float32)
            diff[overlap] = (rendered_depth[overlap] - real_depth[overlap]) * 1000.0
            im3 = axes[3].imshow(diff, cmap="coolwarm", vmin=-50, vmax=50)
            title_err = f"{depth_err_mm:.1f}mm" if depth_err_mm is not None else "N/A"
            axes[3].set_title(f"rendered - real (mm), mean|err|={title_err}")
            axes[3].axis("off")
            plt.colorbar(im3, ax=axes[3], fraction=0.046)

            plt.tight_layout()
            plt.savefig(os.path.join(args.viz_dir, f"{seq}_{frame_id}.png"), dpi=110)
            plt.close(fig)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("RESULTS:", results)


if __name__ == "__main__":
    main()
