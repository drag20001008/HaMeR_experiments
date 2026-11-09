"""
DexYCB 원본 데이터를 HaMeR/WiLoR의 ImageDataset(npz 모드)이 기대하는 형식으로 변환.

산출 npz 필드:
  imgname       (N,)      str   - img_dir 기준 상대경로
  center        (N,2)     float - bbox 중심 (x,y)
  scale         (N,)      float - bbox 크기(raw 픽셀, ImageDataset이 로드 시 자체적으로 /200 함)
  right         (N,)      float - 1.0 = 오른손 (DexYCB는 전부 오른손)
  hand_pose     (N,48)    float - MANO axis-angle (global_orient 3 + hand_pose 45)
  has_hand_pose (N,)      float - 1.0 (전부 GT 있음)
  betas         (N,10)    float - subject별 GT MANO shape
  has_betas     (N,)      float - 1.0
  hand_keypoints_2d (N,21,3) float - (x,y,conf)
  hand_keypoints_3d (N,21,4) float - (x,y,z,conf), 단위: m (DexYCB 원본 그대로, pelvis 빼기 전 절대좌표)
  subject       (N,)      str   - subject id
  fx, fy, px, py (N,)     float - 해당 샘플을 찍은 카메라의 실제 intrinsics (원본 640x480 이미지 기준)
  official_split (N,)     str   - DexYCB 공식 s0(default) split 기준 train/val/test
"""
import os
import glob
import argparse
import yaml
import numpy as np
from tqdm import tqdm


class _SafeLoaderWithTuple(yaml.SafeLoader):
    pass


_SafeLoaderWithTuple.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def yaml_safe_load(f):
    return yaml.load(f, Loader=_SafeLoaderWithTuple)

def load_subject_betas(dexycb_root):
    betas_by_subject = {}
    calib_root = os.path.join(dexycb_root, "calibration")
    for d in os.listdir(calib_root):
        if d.startswith("mano_") and d.endswith("_right"):
            subj = [p for p in d.split("_") if p.startswith("subject-")][0]
            with open(os.path.join(calib_root, d, "mano.yml")) as f:
                y = yaml_safe_load(f)
            betas_by_subject[subj] = np.array(y["betas"], dtype=np.float32)
    return betas_by_subject


def load_camera_intrinsics(dexycb_root):
    """calibration/intrinsics/<serial>_640x480.yml 전부 읽어서 {serial: (fx,fy,px,py)} 반환 (color 카메라 기준)."""
    intr_by_serial = {}
    intr_root = os.path.join(dexycb_root, "calibration", "intrinsics")
    for fname in os.listdir(intr_root):
        if not fname.endswith("_640x480.yml"):
            continue
        serial = fname.split("_640x480.yml")[0]
        with open(os.path.join(intr_root, fname)) as f:
            y = yaml_safe_load(f)
        c = y["color"]
        intr_by_serial[serial] = (
            float(c["fx"]), float(c["fy"]), float(c["ppx"]), float(c["ppy"])
        )
    return intr_by_serial


def s0_split(subject_idx: int, sequence_idx: int) -> str:
    """dex-ycb-toolkit의 DexYCBDataset(setup='s0') 로직 그대로 구현.
    subject_idx: 0~9 (subject-01..10 순서), sequence_idx: 0~99 (subject 내 시퀀스 정렬 순서)."""
    is_val_seq = (sequence_idx % 5 == 4)
    if not is_val_seq:
        return "train"          # subject 0~9, seq%5 != 4
    if subject_idx in (0, 1):
        return "val"            # subject 0,1, seq%5==4
    return "test"                # subject 2~9, seq%5==4


def bbox_from_kp2d(kp2d, pad_ratio=0.3):
    x0, y0 = kp2d[:, 0].min(), kp2d[:, 1].min()
    x1, y1 = kp2d[:, 0].max(), kp2d[:, 1].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = x1 - x0, y1 - y0
    size = max(w, h) * (1 + pad_ratio)
    return cx, cy, size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexycb_root", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--out", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations.npz")
    parser.add_argument("--max_seqs", type=int, default=-1, help="디버깅용: subject당 시퀀스 수 제한 (-1이면 전체)")
    args = parser.parse_args()

    betas_by_subject = load_subject_betas(args.dexycb_root)
    print(f"loaded betas for {len(betas_by_subject)} subjects")

    intr_by_serial = load_camera_intrinsics(args.dexycb_root)
    print(f"loaded intrinsics for {len(intr_by_serial)} cameras")

    imgname_list, center_list, scale_list, right_list = [], [], [], []
    hand_pose_list, betas_list = [], []
    has_hand_pose_list, has_betas_list = [], []
    kp2d_list, kp3d_list, subject_list = [], [], []
    fx_list, fy_list, px_list, py_list = [], [], [], []
    official_split_list = []

    subj_dirs = sorted(glob.glob(os.path.join(args.dexycb_root, "2*-subject-*")))
    n_skipped_invalid = 0
    n_skipped_missing_img = 0

    for subj_idx, subj_dir in enumerate(subj_dirs):
        subj_name = os.path.basename(subj_dir)
        subj_id = "subject-" + subj_name.split("subject-")[1]
        if subj_id not in betas_by_subject:
            print(f"WARNING: no betas found for {subj_id}, skip")
            continue
        betas = betas_by_subject[subj_id]

        seq_dirs = sorted([d for d in glob.glob(os.path.join(subj_dir, "*")) if os.path.isdir(d)])
        if args.max_seqs > 0:
            seq_dirs = seq_dirs[: args.max_seqs]

        for seq_idx, seq_dir in enumerate(tqdm(seq_dirs, desc=subj_name)):
            official_split = s0_split(subj_idx, seq_idx)
            meta_path = os.path.join(seq_dir, "meta.yml")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = yaml_safe_load(f)
            is_right = 1.0 if meta["mano_sides"][0] == "right" else 0.0

            cam_dirs = sorted([d for d in glob.glob(os.path.join(seq_dir, "*")) if os.path.isdir(d)])
            for cam_dir in cam_dirs:
                serial = os.path.basename(cam_dir)
                if serial not in intr_by_serial:
                    continue
                fx, fy, px, py = intr_by_serial[serial]
                label_files = sorted(glob.glob(os.path.join(cam_dir, "labels_*.npz")))
                for lf in label_files:
                    idx = lf.split("labels_")[-1].split(".")[0]
                    color_path = os.path.join(cam_dir, f"color_{idx}.jpg")
                    if not os.path.exists(color_path):
                        n_skipped_missing_img += 1
                        continue

                    d = np.load(lf)
                    joint_2d = d["joint_2d"][0]  # (21,2)
                    joint_3d = d["joint_3d"][0]  # (21,3)
                    if np.all(joint_2d == -1) or np.all(joint_3d == -1):
                        n_skipped_invalid += 1
                        continue

                    pose_m = d["pose_m"][0]  # (51,) = 48 pose + 3 trans
                    hand_pose = pose_m[:48].astype(np.float32)

                    cx, cy, size = bbox_from_kp2d(joint_2d)
                    # ImageDataset이 로드 시 내부적으로 scale/200을 다시 나누므로,
                    # 여기서는 200으로 나누지 않은 raw 픽셀 크기를 그대로 저장한다.
                    scale = size

                    kp2d_conf = np.concatenate(
                        [joint_2d.astype(np.float32), np.ones((21, 1), dtype=np.float32)], axis=1
                    )
                    kp3d_conf = np.concatenate(
                        [joint_3d.astype(np.float32), np.ones((21, 1), dtype=np.float32)], axis=1
                    )

                    rel_path = os.path.relpath(color_path, args.dexycb_root)
                    imgname_list.append(rel_path)
                    center_list.append([cx, cy])
                    scale_list.append(scale)
                    right_list.append(is_right)
                    hand_pose_list.append(hand_pose)
                    betas_list.append(betas)
                    has_hand_pose_list.append(1.0)
                    has_betas_list.append(1.0)
                    kp2d_list.append(kp2d_conf)
                    kp3d_list.append(kp3d_conf)
                    subject_list.append(subj_id)
                    fx_list.append(fx)
                    fy_list.append(fy)
                    px_list.append(px)
                    py_list.append(py)
                    official_split_list.append(official_split)

    print(f"total samples: {len(imgname_list)}")
    print(f"skipped (invalid hand annotation): {n_skipped_invalid}")
    print(f"skipped (missing image file): {n_skipped_missing_img}")

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
        official_split=np.array(official_split_list),
    )

    splits, counts = np.unique(np.array(official_split_list), return_counts=True)
    for s, c in zip(splits, counts):
        print(f"official_split={s}: {c} samples")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
