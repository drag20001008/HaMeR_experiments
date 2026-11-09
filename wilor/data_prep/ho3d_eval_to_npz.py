"""
HO3D_v3 evaluation split을, 별도로 확보한 숨겨진 GT(evaluation_xyz.json,
evaluation_verts.json - evaluation.txt와 같은 순서)를 이용해 WiLoR
ImageDataset(npz 모드) 형식으로 변환. train split용 ho3d_to_npz.py와 동일한
전처리(OpenGL->CV 좌표 변환, raw MANO 순서 -> OpenPose 순서 재배열, 화면 밖 필터)를 적용.

evaluation meta pkl에는 handPose/handBeta가 없으므로(공식적으로 숨겨진 정보),
has_hand_pose=has_betas=0으로 둔다 - 우리 목적(metric-scale/pose 평가)엔
2D/3D keypoints GT만 있으면 충분하다.
"""
import os
import argparse
import pickle
import json
import numpy as np
from tqdm import tqdm

from ho3d_to_npz import gl_to_cv, reorder_to_openpose, MANO_TO_OPENPOSE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ho3d_root", default="/nas2/lait/1000_Members/jaehoon/HO3D_v3")
    parser.add_argument("--eval_xyz_json", default="/workspace/drag1008/evaluation_xyz.json")
    parser.add_argument("--out", default="/nas2/lait/1000_Members/jaehoon/HO3D_v3/ho3d_evaluation_annotations.npz")
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    args = parser.parse_args()

    with open(os.path.join(args.ho3d_root, "evaluation.txt")) as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"evaluation.txt entries: {len(lines)}")

    with open(args.eval_xyz_json) as f:
        xyz_all = json.load(f)
    assert len(xyz_all) == len(lines), f"{len(xyz_all)} vs {len(lines)}"

    imgname_list, center_list, scale_list, right_list = [], [], [], []
    hand_pose_list, betas_list = [], []
    has_hand_pose_list, has_betas_list = [], []
    kp2d_list, kp3d_list, subject_list = [], [], []
    fx_list, fy_list, px_list, py_list = [], [], [], []

    n_skipped_offscreen = 0

    for idx, line in enumerate(tqdm(lines)):
        seq, frame_id = line.split("/")
        meta_path = os.path.join(args.ho3d_root, "evaluation", seq, "meta", f"{frame_id}.pkl")
        rgb_rel = os.path.join("evaluation", seq, "rgb", f"{frame_id}.jpg")
        rgb_path = os.path.join(args.ho3d_root, rgb_rel)
        if not os.path.exists(meta_path) or not os.path.exists(rgb_path):
            continue

        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        cam_mat = np.asarray(meta["camMat"], dtype=np.float32)
        fx, fy = cam_mat[0, 0], cam_mat[1, 1]
        px, py = cam_mat[0, 2], cam_mat[1, 2]

        joints3d_gl = np.asarray(xyz_all[idx], dtype=np.float32)  # (21,3), raw MANO 순서
        joints3d_gl = reorder_to_openpose(joints3d_gl)
        joints3d_cv = gl_to_cv(joints3d_gl)

        Z = joints3d_cv[:, 2]
        if np.any(Z <= 1e-6):
            continue
        u = fx * joints3d_cv[:, 0] / Z + px
        v = fy * joints3d_cv[:, 1] / Z + py

        if np.any((u < 0) | (u >= args.img_w) | (v < 0) | (v >= args.img_h)):
            n_skipped_offscreen += 1
            continue

        # handBoundingBox: [x0,y0,x1,y1] (공식 제공 bbox 그대로 사용)
        x0, y0, x1, y1 = meta["handBoundingBox"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        size = max(x1 - x0, y1 - y0) * 1.3  # train과 동일한 pad_ratio=0.3

        kp2d_conf = np.concatenate([np.stack([u, v], axis=-1).astype(np.float32),
                                     np.ones((21, 1), dtype=np.float32)], axis=1)
        kp3d_conf = np.concatenate([joints3d_cv.astype(np.float32),
                                     np.ones((21, 1), dtype=np.float32)], axis=1)

        imgname_list.append(rgb_rel)
        center_list.append([cx, cy])
        scale_list.append(size)
        right_list.append(1.0)
        hand_pose_list.append(np.zeros(48, dtype=np.float32))
        betas_list.append(np.zeros(10, dtype=np.float32))
        has_hand_pose_list.append(0.0)
        has_betas_list.append(0.0)
        kp2d_list.append(kp2d_conf)
        kp3d_list.append(kp3d_conf)
        subject_list.append(seq)
        fx_list.append(fx)
        fy_list.append(fy)
        px_list.append(px)
        py_list.append(py)

    print(f"total samples: {len(imgname_list)}")
    print(f"skipped (화면 밖): {n_skipped_offscreen}")

    np.savez(
        args.out,
        imgname=np.array(imgname_list),
        center=np.array(center_list, dtype=np.float32),
        scale=np.array(scale_list, dtype=np.float32),
        right=np.array(right_list, dtype=np.float32),
        hand_pose=np.array(hand_pose_list, dtype=np.float32),
        has_hand_pose=np.array(has_hand_pose_list, dtype=np.float32),
        betas=np.array(betas_list, dtype=np.float32),
        has_betas=np.array(has_betas_list, dtype=np.float32),
        hand_keypoints_2d=np.array(kp2d_list, dtype=np.float32),
        hand_keypoints_3d=np.array(kp3d_list, dtype=np.float32),
        subject=np.array(subject_list),
        fx=np.array(fx_list, dtype=np.float32),
        fy=np.array(fy_list, dtype=np.float32),
        px=np.array(px_list, dtype=np.float32),
        py=np.array(py_list, dtype=np.float32),
    )
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
