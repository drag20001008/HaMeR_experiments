"""
Stage 1 체크포인트(baseline / GT-beta)를 재학습 없이 그대로 써서,
21개 관절 최소제곱 solver로 pelvis 절대 3D 위치(depth 포함)를 복원하고
DexYCB 실제 GT와 비교해 metric scale 오차(mm)를 측정하는 평가 스크립트.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset

from solve_absolute_depth import solve_absolute_translation


def split_by_subject(dataset: ImageDataset, val_subjects):
    subjects = dataset.data["subject"]
    val_mask = np.isin(subjects, val_subjects)
    return np.where(val_mask)[0]


def split_by_official(dataset: ImageDataset, split="test"):
    """DexYCB 공식 s0 split(dexycb_to_npz.py가 만든 'official_split' 필드) 기준 분리."""
    official = dataset.data["official_split"]
    return np.where(official == split)[0]


def filter_onscreen(dataset: ImageDataset, idx, img_w=640, img_h=480):
    """DexYCB는 8대 카메라 3D pose를 각 카메라에 재투영해서 라벨을 만드는데,
    이 카메라 각도에서는 손이 화면 밖으로 나가는 프레임도 그대로 라벨에 포함되어 있다
    (dexycb가 -1로 명시한 완전 실패 케이스만 걸러지고, 이 경우는 안 걸러짐).
    관절 중 하나라도 이미지 범위를 벗어나면 제외."""
    kp2d = dataset.data["hand_keypoints_2d"][idx]  # (N,21,3)
    x, y = kp2d[:, :, 0], kp2d[:, :, 1]
    out_of_bounds = ((x < 0) | (x >= img_w) | (y < 0) | (y >= img_h)).any(axis=1)
    return idx[~out_of_bounds]


def pa_mpjpe_single(pred, gt):
    """Procrustes(rotation+scale+translation) 정렬 후 per-joint 평균 거리(mm).
    pred, gt: (21,3) numpy array (m 단위). translation/scale/rotation 오차를 다 제거하고
    순수 관절 상대 배치(각도/비율) 오차만 남긴다."""
    mu_pred = pred.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    X0 = pred - mu_pred
    Y0 = gt - mu_gt
    var_x = (X0 ** 2).sum()
    cov = Y0.T @ X0
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / (var_x + 1e-9)
    X_aligned = scale * (R @ X0.T).T + mu_gt
    return float(np.linalg.norm(X_aligned - gt, axis=-1).mean() * 1000.0)


def pa_mpjpe_batch(pred, gt):
    """pred, gt: (B,21,3) numpy array. 배치 내 샘플별로 pa_mpjpe_single 적용."""
    return np.array([pa_mpjpe_single(pred[b], gt[b]) for b in range(pred.shape[0])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz")
    parser.add_argument("--img_dir", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--beta_conditioning", action="store_true")
    parser.add_argument("--bbox_info", action="store_true",
                         help="학습 시 --bbox_info를 켰던 체크포인트라면 eval도 동일하게 켜야 함")
    parser.add_argument("--real_focal", action="store_true",
                         help="학습 시 --real_focal을 켰던 체크포인트라면 eval도 동일하게 켜야 함")
    parser.add_argument("--split_mode", choices=["subject", "official", "all"], default="official",
                         help="official: DexYCB 공식 s0 split(official_split 필드)의 --official_split 값 사용, "
                              "subject: 기존처럼 --val_subjects로 subject 통째로 분리, "
                              "all: split 없이 npz 전체 사용 (예: HO3D처럼 official_split 필드가 없는 데이터셋)")
    parser.add_argument("--official_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--val_subjects", nargs="+", default=["subject-09", "subject-10"])
    parser.add_argument("--exclude_offscreen", action="store_true",
                         help="관절이 이미지 범위 밖인(카메라 각도상 손이 안 보이는) 샘플 제외")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pelvis_id", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=-1, help="디버깅용, -1이면 전체")
    args = parser.parse_args()

    model, cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.model_config)
    cfg.defrost()
    if "EXPERIMENT" not in cfg:
        from yacs.config import CfgNode as CN
        cfg.EXPERIMENT = CN(new_allowed=True)
    cfg.EXPERIMENT.BETA_CONDITIONING = args.beta_conditioning
    cfg.EXPERIMENT.BBOX_INFO = args.bbox_info
    cfg.EXPERIMENT.REAL_FOCAL = args.real_focal
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    val_dataset_full = ImageDataset(cfg, args.dataset_npz, args.img_dir, train=False, rescale_factor=-1)
    if args.split_mode == "official":
        val_idx = split_by_official(val_dataset_full, args.official_split)
    elif args.split_mode == "all":
        val_idx = np.arange(len(val_dataset_full.data["imgname"]))
    else:
        val_idx = split_by_subject(val_dataset_full, args.val_subjects)
    if args.exclude_offscreen:
        n_before = len(val_idx)
        val_idx = filter_onscreen(val_dataset_full, val_idx)
        print(f"exclude_offscreen: {n_before} -> {len(val_idx)} "
              f"({n_before - len(val_idx)} 제외, {100*(n_before-len(val_idx))/n_before:.1f}%)")
    print(f"split_mode={args.split_mode}, eval samples: {len(val_idx)}")
    val_loader = DataLoader(
        Subset(val_dataset_full, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        return x

    all_errors_mm = []
    all_errors_mm_naive = []
    all_cs_mpjpe = []
    all_pa_mpjpe = []
    for i, batch in enumerate(val_loader):
        if args.max_batches > 0 and i >= args.max_batches:
            break
        batch = to_device(batch)

        with torch.no_grad():
            output = model.forward_step(batch, train=False)

        pred_keypoints_3d = output["pred_keypoints_3d"]  # (B,21,3), raw MANO 출력
        rel_3d = pred_keypoints_3d - pred_keypoints_3d[:, args.pelvis_id:args.pelvis_id + 1, :]
        pred_keypoints_2d = output["pred_keypoints_2d"]  # (B,21,2), crop 패치 픽셀 좌표

        t_pred = solve_absolute_translation(
            pred_keypoints_2d, rel_3d,
            batch["box_center"].float(), batch["box_size"].float(),
            batch["fx"].float(), batch["fy"].float(), batch["px"].float(), batch["py"].float(),
        )

        gt_pelvis_abs = batch["keypoints_3d"][:, args.pelvis_id, :3]  # eval모드라 raw absolute 그대로 보존됨
        error_mm = (t_pred - gt_pelvis_abs).norm(dim=-1) * 1000.0

        # solver 없이, 네트워크 자체 camera head(pred_cam_t, 가짜 고정 focal length 5000 기반) 그대로 썼을 때
        pred_pelvis_naive = pred_keypoints_3d[:, args.pelvis_id, :] + output["pred_cam_t"]
        error_mm_naive = (pred_pelvis_naive - gt_pelvis_abs).norm(dim=-1) * 1000.0
        all_errors_mm_naive.append(error_mm_naive.cpu().numpy())
        all_errors_mm.append(error_mm.cpu().numpy())

        # CS-MPJPE: solver로 구한 절대 translation을 rel_3d에 더해 21개 관절 전체를
        # camera space(alignment 없이)에서 GT와 비교 - translation/scale/pose 오차가 다 섞인 지표
        gt_keypoints_3d_abs = batch["keypoints_3d"][:, :, :3]
        pred_abs = rel_3d + t_pred.unsqueeze(1)
        cs_mpjpe = (pred_abs - gt_keypoints_3d_abs).norm(dim=-1).mean(dim=-1) * 1000.0
        all_cs_mpjpe.append(cs_mpjpe.cpu().numpy())

        # PA-MPJPE: pelvis-relative 구조만 Procrustes(회전+스케일+이동) 정렬 후 비교
        # - translation/scale 오차를 다 제거한, 순수 pose(관절 각도/비율) 오차
        gt_rel_3d = gt_keypoints_3d_abs - gt_keypoints_3d_abs[:, args.pelvis_id:args.pelvis_id + 1, :]
        pa_err = pa_mpjpe_batch(rel_3d.cpu().numpy(), gt_rel_3d.cpu().numpy())
        all_pa_mpjpe.append(pa_err)

        if i % 20 == 0:
            print(f"batch {i}/{len(val_loader)}, running mean error so far: "
                  f"{np.concatenate(all_errors_mm).mean():.1f} mm")

    all_errors_mm = np.concatenate(all_errors_mm)
    all_errors_mm_naive = np.concatenate(all_errors_mm_naive)
    all_cs_mpjpe = np.concatenate(all_cs_mpjpe)
    all_pa_mpjpe = np.concatenate(all_pa_mpjpe)
    print("=" * 50)
    print(f"checkpoint: {args.checkpoint}")
    print(f"beta_conditioning: {args.beta_conditioning}  bbox_info: {args.bbox_info}  real_focal: {args.real_focal}")
    print(f"exclude_offscreen: {args.exclude_offscreen}")
    print(f"N samples: {len(all_errors_mm)}")
    print(f"[translation(solver)]  mean={all_errors_mm.mean():.2f} mm  median={np.median(all_errors_mm):.2f} mm  std={all_errors_mm.std():.2f} mm")
    print(f"[naive, no solver, fake f=5000]  mean={all_errors_mm_naive.mean():.2f} mm  median={np.median(all_errors_mm_naive):.2f} mm  std={all_errors_mm_naive.std():.2f} mm")
    print(f"[CS-MPJPE, solver 절대위치+21관절, align 없음]  mean={all_cs_mpjpe.mean():.2f} mm  median={np.median(all_cs_mpjpe):.2f} mm  std={all_cs_mpjpe.std():.2f} mm")
    print(f"[PA-MPJPE, Procrustes 정렬 후 순수 pose 오차]  mean={all_pa_mpjpe.mean():.2f} mm  median={np.median(all_pa_mpjpe):.2f} mm  std={all_pa_mpjpe.std():.2f} mm")


if __name__ == "__main__":
    main()
