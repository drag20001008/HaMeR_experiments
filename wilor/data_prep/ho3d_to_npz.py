"""
HO3D_v3 (train split, GT annotation 있는 쪽)을 WiLoR ImageDataset(npz 모드)이
기대하는 형식으로 변환. DexYCB용 dexycb_to_npz.py와 동일한 스키마를 맞춘다.

주의: HO3D는 annotation이 OpenGL 좌표계(카메라가 -Z 방향을 봄, Y도 반전)를 쓴다고
README에 명시되어 있음. 표준 CV(OpenCV) 좌표계(카메라가 +Z 방향을 봄)로 변환해줘야
perspective 투영 공식(u=fx*X/Z+cx)이 맞게 동작한다: (X,Y,Z)_cv = (X,-Y,-Z)_gl.
"""
import os
import glob
import argparse
import pickle
import numpy as np
import cv2
from tqdm import tqdm


def gl_to_cv(points):
    """OpenGL 좌표계 -> OpenCV(표준 CV) 좌표계. points: (...,3)"""
    out = points.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


# HO3D의 handJoints3D는 raw MANO 순서(wrist, index(3), middle(3), pinky(3), ring(3),
# thumb(3), thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip)인데, WiLoR
# 모델(wilor/models/mano_wrapper.py의 MANO.forward)의 출력은 OpenPose 순서로
# 재배열되어 나온다. 그 재배열과 똑같은 순서로 GT를 맞춰줘야 관절별 비교가 맞다.
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def reorder_to_openpose(joints21):
    """joints21: (21,3) 또는 (21,) raw MANO 순서 -> OpenPose 순서."""
    return joints21[MANO_TO_OPENPOSE]


def bbox_from_kp2d(kp2d, pad_ratio=0.3):
    x0, y0 = kp2d[:, 0].min(), kp2d[:, 1].min()
    x1, y1 = kp2d[:, 0].max(), kp2d[:, 1].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = x1 - x0, y1 - y0
    size = max(w, h) * (1 + pad_ratio)
    return cx, cy, size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ho3d_root", default="/nas2/lait/1000_Members/jaehoon/HO3D_v3")
    parser.add_argument("--out", default="/nas2/lait/1000_Members/jaehoon/HO3D_v3/ho3d_annotations.npz")
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    args = parser.parse_args()

    with open(os.path.join(args.ho3d_root, "train.txt")) as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"train.txt entries: {len(lines)}")

    imgname_list, center_list, scale_list, right_list = [], [], [], []
    hand_pose_list, betas_list = [], []
    has_hand_pose_list, has_betas_list = [], []
    kp2d_list, kp3d_list, subject_list = [], [], []
    fx_list, fy_list, px_list, py_list = [], [], [], []

    n_skipped_none = 0
    n_skipped_offscreen = 0

    for line in tqdm(lines):
        seq, frame_id = line.split("/")
        meta_path = os.path.join(args.ho3d_root, "train", seq, "meta", f"{frame_id}.pkl")
        rgb_rel = os.path.join("train", seq, "rgb", f"{frame_id}.jpg")
        rgb_path = os.path.join(args.ho3d_root, rgb_rel)
        if not os.path.exists(meta_path) or not os.path.exists(rgb_path):
            continue

        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")

        if meta.get("handPose") is None or meta.get("handJoints3D") is None:
            n_skipped_none += 1
            continue

        hand_pose = np.asarray(meta["handPose"], dtype=np.float32)  # (48,)
        betas = np.asarray(meta["handBeta"], dtype=np.float32)      # (10,)
        joints3d_gl = np.asarray(meta["handJoints3D"], dtype=np.float32)  # (21,3), raw MANO 순서
        joints3d_gl = reorder_to_openpose(joints3d_gl)  # WiLoR 출력과 같은 OpenPose 순서로 맞춤
        cam_mat = np.asarray(meta["camMat"], dtype=np.float32)

        joints3d_cv = gl_to_cv(joints3d_gl)

        fx, fy = cam_mat[0, 0], cam_mat[1, 1]
        px, py = cam_mat[0, 2], cam_mat[1, 2]

        # perspective 투영으로 2D 라벨 생성 (표준 CV 좌표계, Z>0 가정)
        Z = joints3d_cv[:, 2]
        if np.any(Z <= 1e-6):
            continue
        u = fx * joints3d_cv[:, 0] / Z + px
        v = fy * joints3d_cv[:, 1] / Z + py
        joint_2d = np.stack([u, v], axis=-1)

        # DexYCB 때 배운 교훈: 관절이 이미지 범위 밖이면 처음부터 제외
        if np.any((u < 0) | (u >= args.img_w) | (v < 0) | (v >= args.img_h)):
            n_skipped_offscreen += 1
            continue

        cx, cy, size = bbox_from_kp2d(joint_2d)

        kp2d_conf = np.concatenate([joint_2d.astype(np.float32), np.ones((21, 1), dtype=np.float32)], axis=1)
        kp3d_conf = np.concatenate([joints3d_cv.astype(np.float32), np.ones((21, 1), dtype=np.float32)], axis=1)

        imgname_list.append(rgb_rel)
        center_list.append([cx, cy])
        scale_list.append(size)
        right_list.append(1.0)  # HO3D는 전부 오른손
        hand_pose_list.append(hand_pose)
        betas_list.append(betas)
        has_hand_pose_list.append(1.0)
        has_betas_list.append(1.0)
        kp2d_list.append(kp2d_conf)
        kp3d_list.append(kp3d_conf)
        subject_list.append(seq)
        fx_list.append(fx)
        fy_list.append(fy)
        px_list.append(px)
        py_list.append(py)

    print(f"total samples: {len(imgname_list)}")
    print(f"skipped (None annotation): {n_skipped_none}")
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
