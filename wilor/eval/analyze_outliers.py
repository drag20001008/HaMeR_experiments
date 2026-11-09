"""
solve_absolute_translation 결과 중 오차가 유독 큰 이상치 샘플들을 찾아서
어떤 subject/box_size/s 값에서 발생하는지 확인하는 진단 스크립트.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset

from solve_absolute_depth import solve_absolute_translation


def split_by_official(dataset, split="test"):
    official = dataset.data["official_split"]
    return np.where(official == split)[0]


def pa_mpjpe(pred, gt):
    """Procrustes(rotation+scale+translation) 정렬 후 per-joint 평균 거리(mm).
    pred, gt: (21,3) numpy array, 같은 단위(m). translation/scale 오차를 다 제거하고
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz")
    parser.add_argument("--img_dir", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--beta_conditioning", action="store_true")
    parser.add_argument("--bbox_info", action="store_true")
    parser.add_argument("--real_focal", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pelvis_id", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=30)
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
    val_idx = split_by_official(val_dataset_full, "test")
    subjects_all = val_dataset_full.data["subject"][val_idx]
    imgnames_all = val_dataset_full.data["imgname"][val_idx]
    print(f"eval samples: {len(val_idx)}")
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

    records = []
    offset = 0
    for i, batch in enumerate(val_loader):
        if args.max_batches > 0 and i >= args.max_batches:
            break
        bsz = batch["box_size"].shape[0]
        batch = to_device(batch)

        with torch.no_grad():
            output = model.forward_step(batch, train=False)

        pred_keypoints_3d = output["pred_keypoints_3d"]
        rel_3d = pred_keypoints_3d - pred_keypoints_3d[:, args.pelvis_id:args.pelvis_id + 1, :]
        pred_keypoints_2d = output["pred_keypoints_2d"]
        pred_cam = output["pred_cam"]  # (B,3) = s, tx, ty

        t_pred = solve_absolute_translation(
            pred_keypoints_2d, rel_3d,
            batch["box_center"].float(), batch["box_size"].float(),
            batch["fx"].float(), batch["fy"].float(), batch["px"].float(), batch["py"].float(),
        )
        gt_pelvis_abs = batch["keypoints_3d"][:, args.pelvis_id, :3]
        error_mm = (t_pred - gt_pelvis_abs).norm(dim=-1) * 1000.0

        gt_keypoints_3d = batch["keypoints_3d"][:, :, :3]
        gt_rel_3d = gt_keypoints_3d - gt_keypoints_3d[:, args.pelvis_id:args.pelvis_id + 1, :]
        rel_3d_np = rel_3d.cpu().numpy()
        gt_rel_3d_np = gt_rel_3d.cpu().numpy()

        for b in range(bsz):
            pa_error = pa_mpjpe(rel_3d_np[b], gt_rel_3d_np[b])
            records.append({
                "idx": offset + b,
                "subject": subjects_all[offset + b],
                "imgname": imgnames_all[offset + b],
                "error_mm": error_mm[b].item(),
                "pa_mpjpe_mm": pa_error,
                "box_size": batch["box_size"][b].item(),
                "s": pred_cam[b, 0].item(),
                "fx": batch["fx"][b].item(),
            })
        offset += bsz

    errors = np.array([r["error_mm"] for r in records])
    pa_errors = np.array([r["pa_mpjpe_mm"] for r in records])
    print(f"N={len(errors)}  mean={errors.mean():.2f}  median={np.median(errors):.2f}  "
          f"p90={np.percentile(errors,90):.2f}  p99={np.percentile(errors,99):.2f}  max={errors.max():.2f}")
    print(f"PA-MPJPE  mean={pa_errors.mean():.2f}  median={np.median(pa_errors):.2f}  "
          f"p90={np.percentile(pa_errors,90):.2f}  p99={np.percentile(pa_errors,99):.2f}  max={pa_errors.max():.2f}")
    print(f"corr(error_mm, pa_mpjpe) = {np.corrcoef(errors, pa_errors)[0,1]:.3f}")

    records_sorted = sorted(records, key=lambda r: -r["error_mm"])
    print(f"\n=== Top {args.top_k} 오차 샘플 (translation 오차 기준 정렬) ===")
    for r in records_sorted[:args.top_k]:
        print(f"error={r['error_mm']:9.1f}mm  pa_mpjpe={r['pa_mpjpe_mm']:7.1f}mm  subject={r['subject']}  "
              f"box_size={r['box_size']:.1f}  s={r['s']:.4f}  fx={r['fx']:.1f}  img={r['imgname']}")

    print(f"\n=== subject별 mean/median error / PA-MPJPE ===")
    subj_arr = np.array([r["subject"] for r in records])
    for subj in sorted(set(subj_arr.tolist())):
        mask = subj_arr == subj
        e = errors[mask]
        pe = pa_errors[mask]
        print(f"{subj}: N={mask.sum():5d}  error mean={e.mean():8.2f}mm median={np.median(e):7.2f}mm  "
              f"| PA-MPJPE mean={pe.mean():6.2f}mm median={np.median(pe):6.2f}mm")


if __name__ == "__main__":
    main()
