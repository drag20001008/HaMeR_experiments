"""
AGILE 손 초기화 재현 (WiLoR 기반, 수정된 ICP), DexYCB test. shard 병렬 실행용 (hamer env: pytorch3d).

AGILE (Sec 3.2): WiLoR MANO(예측 beta, theta, R) -> R 고정, ICP로 scale s_h (+임시 translation) 추정
                 -> s_h 적용한 3D joint + WiLoR 2D + K 로 PnP -> 최종 T_h.  (+ 추적 단계의 L_joint 정제 = GN)
이전 구현(hamer/analyze_translation_axiswise_dexycb_original_agile.py)의 버그: ICP 초기 t=0(카메라 원점)
  -> 대응이 몇 점에 몰려 s가 0으로 붕괴 -> T≈0.
논문 그대로 버전 (임의 장치 없음):
  - ICP: mesh 전체 정점 -> 가장 가까운 cloud 점 (model-to-scan), R 고정, s와 t만 닫힌 형태로 반복 추정, scale 제한 없음
  - 논문에 명시가 없어 정한 최소한의 것: 초기값 s=1, t = WiLoR 2D + K 선형 PnP 해 / trimming 70% (Trimmed ICP) /
    최대 30회, 수렴 시 중단 / PnP = 선형 최소제곱 / L_joint 최적화 = Gauss-Newton
  - 손 마스크: SAM3 결과 그대로 (sam3_dexycb_hand_masks.py, shard k <-> 이 스크립트 shard k 동일 샘플).
    SAM3 마스크 없음 또는 cloud < 100점이면 ICP 생략, s=1.
depth 소스 두 가지 (둘 다 SAM3 손 마스크 안의 픽셀로 point cloud 생성):
  sensor : DexYCB aligned depth (RealSense)
  gt     : GT MANO mesh 렌더링 depth (완벽한 depth 가정)
"""
import argparse
import os

import numpy as np
import smplx
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from torch.utils.data import DataLoader

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset
from eval_metric_scale import split_by_official
from eval_dexycb_pnp_gtbeta import solve_T, gn_refine
from extract_depth_anchor import ROOT, TIPS, REORDER, WithDepth

SOURCES = ("sensor", "gt")


def icp_scale(p, cloud, t0, iters=30, trim=0.7, s_lim=None, tol=1e-5):
    """p: (V,3) 보이는 정점 (손목 기준 상대좌표), cloud: (C,3). R 고정, s와 t만. 반환 (s, t, n_iter)."""
    s = torch.ones((), device=p.device, dtype=p.dtype)
    t = t0.clone()
    k = max(20, int(trim * p.shape[0]))
    for it in range(iters):
        nn_dist, nn_idx = torch.cdist((s * p + t)[None], cloud[None])[0].min(1)
        keep = torch.topk(-nn_dist, min(k, p.shape[0])).indices
        pp, qq = p[keep], cloud[nn_idx[keep]]
        pm, qm = pp.mean(0), qq.mean(0)
        s_new = ((pp - pm) * (qq - qm)).sum() / ((pp - pm) ** 2).sum().clamp_min(1e-12)
        if s_lim is not None:
            s_new = s_new.clamp(*s_lim)
        t_new = qm - s_new * pm
        done = (s_new - s).abs() < tol and (t_new - t).norm() < tol
        s, t = s_new, t_new
        if done:
            break
    return s, t, it + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", default=f"{ROOT}/wilor_annotations_v3.npz")
    ap.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    ap.add_argument("--checkpoint", default="./pretrained_models/wilor_final.ckpt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--max_cloud", type=int, default=4000, help="point cloud 최대 점 수 (균등 subsample)")
    ap.add_argument("--out_dir", default="./agile_extract")
    ap.add_argument("--sam3_dir", default=f"{ROOT}/sam3_hand_masks")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(args.shard)

    model, cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.model_config)
    model = model.to(dev).eval()
    gt_layers = {s: smplx.MANO(model_path="mano_data_dexycb", is_rhand=(s == "right"), use_pca=False,
                               num_pca_comps=45, flat_hand_mean=False).to(dev) for s in ("right", "left")}
    gt_comps = {s: torch.tensor(np.asarray(L.np_hand_components), dtype=torch.float32, device=dev)
                for s, L in gt_layers.items()}
    faces = {"pred": torch.as_tensor(np.asarray(model.mano.faces, dtype=np.int64), device=dev),
             "right": torch.as_tensor(np.asarray(gt_layers["right"].faces, dtype=np.int64), device=dev),
             "left": torch.as_tensor(np.asarray(gt_layers["left"].faces, dtype=np.int64), device=dev)}

    ds_full = ImageDataset(cfg, args.dataset_npz, ROOT, train=False, rescale_factor=-1)
    idx_all = split_by_official(ds_full, "test")[args.shard::args.n_shards]
    sm = np.load(os.path.join(args.sam3_dir, f"shard{args.shard:02d}.npz"))
    assert args.n_shards == 8 and (sm["idx"] == idx_all).all(), "SAM3 shard와 샘플 순서 불일치"
    sam3_pos = {int(i): k for k, i in enumerate(sm["idx"])}
    sam3_bits, sam3_ok = sm["mask"], sm["ok"]
    print(f"shard {args.shard}/{args.n_shards}: {len(idx_all)} samples", flush=True)
    loader = DataLoader(WithDepth(ds_full, idx_all), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    H, W = 480, 640
    raster = MeshRasterizer(raster_settings=RasterizationSettings(image_size=(H, W), blur_radius=0.0,
                                                                  faces_per_pixel=1, bin_size=0))
    uu, vv = torch.meshgrid(torch.arange(W, device=dev, dtype=torch.float32),
                            torch.arange(H, device=dev, dtype=torch.float32), indexing="xy")

    acc = {k: [] for k in ("idx", "right", "onscreen", "gt_j", "pnp_j", "sam3_ok")}
    for src in SOURCES:
        for k in ("s", "icp_ok", "n_cloud", "n_iter", "agile_j", "agile_gn_j"):
            acc[f"{src}_{k}"] = []

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    for bi, batch in enumerate(loader):
        if 0 < args.max_batches <= bi:
            break
        batch = to_dev(batch)
        with torch.no_grad():
            out = model.forward_step(batch, train=False)
            B = out["pred_cam"].shape[0]
            right = batch["right"].float()
            is_left = right < 0.5
            sign = (2 * right - 1).view(B, 1)

            def unm(x):
                x = x.clone()
                x[..., 0] = x[..., 0] * sign
                return x

            fx, fy, px, py = [batch[k].float() for k in ("fx", "fy", "px", "py")]
            bc, bsz = batch["box_center"].float(), batch["box_size"].float()
            gt3d = unm(batch["keypoints_3d"][..., :3].float())
            gt2d = batch["orig_keypoints_2d"][..., :2].float()
            onscreen = ((gt2d[..., 0] >= 0) & (gt2d[..., 0] < W) & (gt2d[..., 1] >= 0) & (gt2d[..., 1] < H)).all(1)

            # WiLoR 그대로 (예측 beta): 손목 기준 상대 joint / vertex, 원래 카메라 좌표계로 unmirror
            j = out["pred_keypoints_3d"].float()
            v = out["pred_vertices"].float()
            rel_j, rel_v = unm(j - j[:, :1]), unm(v - j[:, :1])
            uv_w = bc[:, None] + unm(out["pred_keypoints_2d"].float()) * bsz.view(B, 1, 1)
            T0 = solve_T(rel_j, uv_w, fx, fy, px, py)  # ICP 초기값 겸 s=1 PnP 결과

            # 카메라 설정 + 렌더링 (초기 위치의 예측 mesh -> 가시 정점, GT mesh -> gt depth)
            Km = torch.zeros(B, 3, 3, device=dev)
            Km[:, 0, 0], Km[:, 1, 1], Km[:, 0, 2], Km[:, 1, 2], Km[:, 2, 2] = fx, fy, px, py, 1
            cams = cameras_from_opencv_projection(torch.eye(3, device=dev).expand(B, 3, 3), torch.zeros(B, 3, device=dev),
                                                  Km, torch.tensor([[H, W]], device=dev).expand(B, 2))
            pmg = batch["pose_m"].float()
            betas_gt = batch["mano_params"]["betas"].float()
            gv = {}
            for sd, Lr in gt_layers.items():
                gv[sd] = Lr(global_orient=pmg[:, :3], hand_pose=pmg[:, 3:48] @ gt_comps[sd], betas=betas_gt,
                            transl=pmg[:, 48:51]).vertices
            gt_v = torch.where(is_left.view(B, 1, 1), gv["left"], gv["right"]).float()
            zb_g = raster(Meshes(verts=list(gt_v), faces=[faces["left"] if l else faces["right"] for l in is_left.tolist()]),
                          cameras=cams).zbuf[..., 0]

            # SAM3 손 마스크 (결과 그대로)
            kk = [sam3_pos[int(i)] for i in batch["idx"].tolist()]
            hand = torch.from_numpy(np.unpackbits(sam3_bits[kk], axis=1)[:, :H * W].reshape(B, H, W).astype(bool)).to(dev)
            s3ok = sam3_ok[kk]
            depth = {"sensor": batch["depth"].float(), "gt": torch.where(zb_g > 0, zb_g, torch.zeros_like(zb_g))}
            res = {src: {k: [] for k in ("s", "ok", "nc", "ni", "Sj")} for src in SOURCES}
            for b in range(B):
                p = rel_v[b]  # mesh 전체 정점 (논문 그대로)
                for src in SOURCES:
                    D = depth[src][b]
                    m = hand[b] & (D > 0)
                    Z = D[m]
                    cloud = torch.stack([(uu[m] - px[b]) * Z / fx[b], (vv[m] - py[b]) * Z / fy[b], Z], -1)
                    if cloud.shape[0] > args.max_cloud:
                        cloud = cloud[torch.randperm(cloud.shape[0], device=dev, generator=g)[:args.max_cloud]]
                    ok = bool(s3ok[b]) and cloud.shape[0] >= 100
                    if ok:
                        s, _, ni = icp_scale(p, cloud, T0[b])
                    else:
                        s, ni = torch.ones((), device=dev), 0
                    res[src]["s"].append(float(s)); res[src]["ok"].append(ok)
                    res[src]["nc"].append(cloud.shape[0]); res[src]["ni"].append(ni)
            for src in SOURCES:
                s = torch.tensor(res[src]["s"], device=dev).view(B, 1, 1)
                Sj = rel_j * s  # scale 적용한 손목 기준 3D joint
                T = solve_T(Sj, uv_w, fx, fy, px, py)
                Tg = gn_refine(Sj, uv_w, fx, fy, px, py, T)
                acc[f"{src}_s"].append(s.view(B).cpu().numpy())
                acc[f"{src}_icp_ok"].append(np.array(res[src]["ok"]))
                acc[f"{src}_n_cloud"].append(np.array(res[src]["nc"]))
                acc[f"{src}_n_iter"].append(np.array(res[src]["ni"]))
                acc[f"{src}_agile_j"].append((Sj + T[:, None]).cpu().numpy())
                acc[f"{src}_agile_gn_j"].append((Sj + Tg[:, None]).cpu().numpy())
            acc["idx"].append(batch["idx"].cpu().numpy())
            acc["right"].append(right.cpu().numpy())
            acc["onscreen"].append(onscreen.cpu().numpy())
            acc["gt_j"].append(gt3d.cpu().numpy())
            acc["pnp_j"].append((rel_j + T0[:, None]).cpu().numpy())
            acc["sam3_ok"].append(np.asarray(s3ok))
        if bi % 20 == 0:
            ss = np.concatenate(acc["sensor_s"]); sg = np.concatenate(acc["gt_s"])
            print(f"batch {bi}/{len(loader)}  ICP scale median sensor {np.median(ss):.3f} gt {np.median(sg):.3f}", flush=True)

    R = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(os.path.join(args.out_dir, f"shard{args.shard:02d}.npz"), **R)
    print(f"saved shard {args.shard}: N={len(R['idx'])}", flush=True)


if __name__ == "__main__":
    main()
