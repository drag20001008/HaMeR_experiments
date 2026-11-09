"""
DexYCB(official s0 test) + pretrained WiLoR: GT beta + WiLoR theta/R + WiLoR 2D keypoint로
translation만 PnP(closed-form DLT)로 풀어 camera-space 손을 복원하고,
CS-MPJPE / 손목 translation 축별 오차 / 2D keypoint 시각화를 만든다.

모든 비교는 원본 이미지 좌표계 + 실제 K 기준. get_example이 왼손을 flip해서 네트워크에 넣으므로
3D 예측/GT, crop 2D, pred_cam tx 모두 x 부호를 되돌린 뒤(unmirror) 비교한다.

방법:
  A_demo_fakef       : demo.py 방식. pred_cam -> cam_crop_to_full (f=5000/256*max(W,H), pp=이미지 중심) (참고용)
  B_cam_realK        : pred_cam(weak-perspective)을 실제 fx,fy,px,py로 full image 변환
  C_pnp_wbeta        : WiLoR beta joint + WiLoR 2D -> PnP
  C_pnp_wbeta_nounm  : C와 같지만 왼손 unmirror 없이 (기존 eval_metric_scale.py 방식)
  D_pnp_gtbeta       : GT beta + WiLoR theta/R joint + WiLoR 2D -> PnP closed-form   (메인)
  D_pnp_gtbeta_gn    : D + Gauss-Newton (algebraic error 대신 진짜 픽셀 reprojection error 최소화)
  E_pnp_gtbeta_gt2d  : D에서 2D만 GT로 (oracle 2D)
  E_pnp_gtbeta_gt2d_gn : E + Gauss-Newton
  *_lfix             : 왼손 샘플만 shapedirs x성분 부호를 뒤집은 MANO로 GT beta joint 생성.
                       DexYCB(manopth)가 쓰는 MANO_LEFT.pkl은 shapedirs x가 오른손과 동일(미반전)한 알려진 버그가 있어,
                       오른손 MANO를 좌우반전해 쓰면 GT beta 형상이 DexYCB 피팅과 달라지는지 확인용. 오른손은 동일.

추가 지표:
  rp_fit / rp_gt : 최종 3D를 실제 K로 투영했을 때 WiLoR 2D / GT 2D와의 평균 픽셀 거리
  ray 분해       : 손목 오차 e를 GT 손목 시선 방향 r=p_gt/|p_gt| 기준으로
                   e_ray = e·r (+면 GT보다 멀리), e_perp = |e - e_ray r|
  depth 구간     : GT 손목 z 기준 구간별 CS-MPJPE / 손목 z / ray 오차
"""
import argparse
import copy
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset
from eval_metric_scale import split_by_official

METHODS = ["A_demo_fakef", "B_cam_realK", "C_pnp_wbeta", "C_pnp_wbeta_nounm", "D_pnp_gtbeta", "D_pnp_gtbeta_gn",
           "E_pnp_gtbeta_gt2d", "E_pnp_gtbeta_gt2d_gn",
           "D_pnp_gtbeta_lfix", "D_pnp_gtbeta_gn_lfix", "E_pnp_gtbeta_gt2d_lfix", "E_pnp_gtbeta_gt2d_gn_lfix"]
DEPTH_BINS = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def solve_T(rel, uv, fx, fy, px, py):
    """R 고정, translation만: fx*tx - (u-px)*tz = (u-px)*z - fx*x (v도 동일). 21x2 과결정 선형 최소제곱."""
    B, N, _ = rel.shape
    x, y, z = rel.double().unbind(-1)
    u, v = uv.double().unbind(-1)
    fx_, fy_, px_, py_ = [a.double().view(B, 1) for a in (fx, fy, px, py)]
    zero = torch.zeros_like(u)
    A = torch.cat([torch.stack([fx_.expand(-1, N), zero, -(u - px_)], -1),
                   torch.stack([zero, fy_.expand(-1, N), -(v - py_)], -1)], 1)
    b = torch.cat([(u - px_) * z - fx_ * x, (v - py_) * z - fy_ * y], 1).unsqueeze(-1)
    return torch.linalg.lstsq(A, b).solution.squeeze(-1).float()


def gn_refine(rel, uv, fx, fy, px, py, T, iters=10):
    """픽셀 reprojection error를 Gauss-Newton으로 최소화 (T 3개만)."""
    rel, uv, T = rel.double(), uv.double(), T.double()
    fx, fy, px, py = [a.double().view(-1, 1) for a in (fx, fy, px, py)]
    eye = 1e-9 * torch.eye(3, dtype=T.dtype, device=T.device)
    for _ in range(iters):
        X, Y, Z = (rel + T[:, None]).unbind(-1)
        zero = torch.zeros_like(X)
        r = torch.cat([fx * X / Z + px - uv[..., 0], fy * Y / Z + py - uv[..., 1]], 1).unsqueeze(-1)
        J = torch.cat([torch.stack([fx / Z, zero, -fx * X / Z ** 2], -1),
                       torch.stack([zero, fy / Z, -fy * Y / Z ** 2], -1)], 1)
        Jt = J.transpose(1, 2)
        T = T - torch.linalg.solve(Jt @ J + eye, Jt @ r).squeeze(-1)
    return T.float()


def project(P, fx, fy, px, py):
    return torch.stack([fx.view(-1, 1) * P[..., 0] / P[..., 2] + px.view(-1, 1),
                        fy.view(-1, 1) * P[..., 1] / P[..., 2] + py.view(-1, 1)], -1)


def axis_stats(e):
    """e: (N,3) mm, pred - gt"""
    out = {}
    for i, ax in enumerate("xyz"):
        a = e[:, i]
        out[ax] = dict(bias=float(a.mean()), std=float(a.std()), mae=float(np.abs(a).mean()),
                       rmse=float(np.sqrt((a ** 2).mean())), median_abs=float(np.median(np.abs(a))))
    n = np.linalg.norm(e, axis=1)
    out["norm"] = dict(mean=float(n.mean()), median=float(np.median(n)))
    return out


def ray_split(e, p_gt):
    """e, p_gt: (N,3). 시선 방향 성분(부호, +면 멀리)과 수직 성분 크기."""
    r = p_gt / np.linalg.norm(p_gt, axis=1, keepdims=True)
    e_ray = (e * r).sum(1)
    e_perp = np.linalg.norm(e - e_ray[:, None] * r, axis=1)
    return e_ray, e_perp


def method_stats(R, m, msk):
    cs = R[f"cs_{m}"][msk]
    e = R[f"wrist_{m}"][msk]
    e_ray, e_perp = ray_split(e, R["gt_wrist"][msk])
    return {"N": int(msk.sum()),
            "CS_MPJPE_mean": float(cs.mean()), "CS_MPJPE_median": float(np.median(cs)),
            "wrist": axis_stats(e),
            "ray": dict(bias=float(e_ray.mean()), mae=float(np.abs(e_ray).mean()),
                        median_abs=float(np.median(np.abs(e_ray))),
                        perp_mean=float(e_perp.mean()), perp_median=float(np.median(e_perp))),
            "rp_fit_px": dict(mean=float(R[f"rpfit_{m}"][msk].mean()), median=float(np.median(R[f"rpfit_{m}"][msk]))),
            "rp_gt_px": dict(mean=float(R[f"rpgt_{m}"][msk].mean()), median=float(np.median(R[f"rpgt_{m}"][msk])))}


def draw_hand(img, kp, color, thick):
    for a, b in BONES:
        cv2.line(img, tuple(np.round(kp[a]).astype(int)), tuple(np.round(kp[b]).astype(int)), color, thick, cv2.LINE_AA)
    for p in kp:
        cv2.circle(img, tuple(np.round(p).astype(int)), thick + 1, color, -1, cv2.LINE_AA)


def make_tile(rec, tile=320):
    img = cv2.imread(rec["imgname"])
    cx, cy = rec["box_center"]
    half = 0.65 * rec["box_size"]
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    # 이미지 밖은 검정으로 패딩 후 crop
    pad = side
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    s = tile / side
    offs = np.array([x0, y0], dtype=np.float32)
    crop = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    crop = cv2.resize(crop, (tile, tile))
    draw_hand(crop, (rec["gt2d"] - offs) * s, (0, 200, 0), 2)       # GT: green
    draw_hand(crop, (rec["wilor2d"] - offs) * s, (0, 0, 255), 1)    # WiLoR 2D: red
    draw_hand(crop, (rec["reprojD"] - offs) * s, (255, 128, 0), 1)  # D 결과 reprojection: blue
    ex, ey, ez = rec["errD"]
    lines = [f"#{rec['idx']} {'R' if rec['right'] > 0.5 else 'L'}  2D {rec['err2d']:.1f}px",
             f"D wrist dx{ex:+.0f} dy{ey:+.0f} dz{ez:+.0f}mm"]
    for i, t in enumerate(lines):
        cv2.putText(crop, t, (5, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, t, (5, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def save_grid(recs, path, cols=6, tile=320):
    tiles = [make_tile(r, tile) for r in recs]
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    grid = np.concatenate(rows, 0)
    legend = np.full((30, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(legend, "green: GT 2D   red: WiLoR 2D (crop->full)   blue: reprojection of D_lfix (GT beta + PnP, real K)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.concatenate([legend, grid], 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz")
    parser.add_argument("--img_dir", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--checkpoint", default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--official_split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1, help="test split에서 k개마다 1개만 사용 (빠른 점검용)")
    parser.add_argument("--n_viz_random", type=int, default=18)
    parser.add_argument("--n_viz_worst", type=int, default=12)
    parser.add_argument("--out_dir", default="./eval_dexycb_pnp_gtbeta_out")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    mano_xflip = copy.deepcopy(model.mano)
    mano_xflip.shapedirs[:, 0, :] *= -1

    ds_full = ImageDataset(cfg, args.dataset_npz, args.img_dir, train=False, rescale_factor=-1)
    idx_all = split_by_official(ds_full, args.official_split)[::args.stride]
    print(f"split={args.official_split}, samples={len(idx_all)}", flush=True)
    loader = DataLoader(Subset(ds_full, idx_all), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    acc = {k: [] for k in ["idx", "right", "onscreen", "err2d", "err2d_nounm", "gt_reproj_err", "gt_wrist",
                           "ra_wbeta", "ra_gtbeta", "ra_gtbeta_lfix", "wilor2d", "gt2d", "reprojD", "box_center", "box_size"]}
    for pre in ("wrist", "cs", "rpfit", "rpgt"):
        acc.update({f"{pre}_{m}": [] for m in METHODS})
    imgnames = []

    for bi, batch in enumerate(loader):
        if 0 < args.max_batches <= bi:
            break
        imgnames += list(batch["imgname"])
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, dict):
                return {k: to_device(v) for k, v in x.items()}
            return x
        batch = to_device(batch)
        with torch.no_grad():
            out = model.forward_step(batch, train=False)
            B = out["pred_cam"].shape[0]
            sign = (2 * batch["right"].float() - 1).view(B, 1)

            def unm(x):
                x = x.clone()
                x[..., 0] = x[..., 0] * sign
                return x

            fx, fy, px, py = [batch[k].float() for k in ("fx", "fy", "px", "py")]
            bc = batch["box_center"].float()
            bsz = batch["box_size"].float()
            W, H = batch["img_size"][:, 0].float(), batch["img_size"][:, 1].float()

            # GT (flip된 3D GT를 원래 카메라 좌표계로 되돌림), 2D GT는 원본 좌표 그대로
            gt3d = unm(batch["keypoints_3d"][..., :3].float())
            gt2d = batch["orig_keypoints_2d"][..., :2].float()
            onscreen = ((gt2d[..., 0] >= 0) & (gt2d[..., 0] < W[:, None]) &
                        (gt2d[..., 1] >= 0) & (gt2d[..., 1] < H[:, None])).all(1)

            # WiLoR joint (예측 beta) / GT beta + WiLoR theta,R 로 다시 만든 joint
            pm = out["pred_mano_params"]
            go = pm["global_orient"].reshape(B, -1, 3, 3)
            hp = pm["hand_pose"].reshape(B, -1, 3, 3)
            j_w = out["pred_keypoints_3d"].float()
            j_g = model.mano(global_orient=go, hand_pose=hp, betas=batch["mano_params"]["betas"].float(),
                             pose2rot=False).joints.float()
            rel_w_m = j_w - j_w[:, :1]
            rel_w = unm(rel_w_m)
            rel_g = unm(j_g - j_g[:, :1])
            j_gx = mano_xflip(global_orient=go, hand_pose=hp, betas=batch["mano_params"]["betas"].float(),
                              pose2rot=False).joints.float()
            is_left = (batch["right"] < 0.5).view(B, 1, 1)
            rel_gl = torch.where(is_left, unm(j_gx - j_gx[:, :1]), rel_g)

            # crop 정규화 2D([-0.5,0.5], = crop pixel/256 - 0.5) -> 원본 이미지 좌표
            kp_n = out["pred_keypoints_2d"].float()
            uv_w = bc[:, None] + unm(kp_n) * bsz.view(B, 1, 1)
            uv_w_nounm = bc[:, None] + kp_n * bsz.view(B, 1, 1)

            preds = {}
            # A: demo.py (가짜 focal, principal point = 이미지 중심)
            pc = out["pred_cam"].float().clone()
            pc[:, 1] = pc[:, 1] * sign.view(B)
            bs = bsz * pc[:, 0] + 1e-9
            f_fake = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * torch.maximum(W, H)
            T_A = torch.stack([2 * (bc[:, 0] - W / 2) / bs + pc[:, 1],
                               2 * (bc[:, 1] - H / 2) / bs + pc[:, 2], 2 * f_fake / bs], -1)
            preds["A_demo_fakef"] = unm(j_w) + T_A[:, None]
            # B: 같은 weak-perspective cam(s,tx,ty)을 실제 K로 해석
            tz = 2 * fx / bs
            T_B = torch.stack([pc[:, 1] + (bc[:, 0] - px) * tz / fx, pc[:, 2] + (bc[:, 1] - py) * tz / fy, tz], -1)
            preds["B_cam_realK"] = unm(j_w) + T_B[:, None]
            # C: WiLoR beta + WiLoR 2D
            preds["C_pnp_wbeta"] = rel_w + solve_T(rel_w, uv_w, fx, fy, px, py)[:, None]
            # C_nounm: flip 좌표계 그대로 (기존 eval 방식) -> 비교를 위해 마지막에만 unmirror
            T_Cn = solve_T(rel_w_m, uv_w_nounm, fx, fy, px, py)
            preds["C_pnp_wbeta_nounm"] = unm(rel_w_m + T_Cn[:, None])
            # D: GT beta + WiLoR theta/R + WiLoR 2D
            T_D = solve_T(rel_g, uv_w, fx, fy, px, py)
            preds["D_pnp_gtbeta"] = rel_g + T_D[:, None]
            preds["D_pnp_gtbeta_gn"] = rel_g + gn_refine(rel_g, uv_w, fx, fy, px, py, T_D)[:, None]
            # E: oracle GT 2D
            T_E = solve_T(rel_g, gt2d, fx, fy, px, py)
            preds["E_pnp_gtbeta_gt2d"] = rel_g + T_E[:, None]
            preds["E_pnp_gtbeta_gt2d_gn"] = rel_g + gn_refine(rel_g, gt2d, fx, fy, px, py, T_E)[:, None]
            T_Dl = solve_T(rel_gl, uv_w, fx, fy, px, py)
            preds["D_pnp_gtbeta_lfix"] = rel_gl + T_Dl[:, None]
            preds["D_pnp_gtbeta_gn_lfix"] = rel_gl + gn_refine(rel_gl, uv_w, fx, fy, px, py, T_Dl)[:, None]
            T_El = solve_T(rel_gl, gt2d, fx, fy, px, py)
            preds["E_pnp_gtbeta_gt2d_lfix"] = rel_gl + T_El[:, None]
            preds["E_pnp_gtbeta_gt2d_gn_lfix"] = rel_gl + gn_refine(rel_gl, gt2d, fx, fy, px, py, T_El)[:, None]

            for m, P in preds.items():
                acc[f"wrist_{m}"].append(((P[:, 0] - gt3d[:, 0]) * 1000).cpu().numpy())
                acc[f"cs_{m}"].append(((P - gt3d).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
                uv_P = project(P, fx, fy, px, py)
                acc[f"rpfit_{m}"].append((uv_P - uv_w).norm(dim=-1).mean(-1).cpu().numpy())
                acc[f"rpgt_{m}"].append((uv_P - gt2d).norm(dim=-1).mean(-1).cpu().numpy())
            acc["gt_wrist"].append(gt3d[:, 0].cpu().numpy())

            gt_rel = gt3d - gt3d[:, :1]
            acc["ra_wbeta"].append(((rel_w - gt_rel).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
            acc["ra_gtbeta"].append(((rel_g - gt_rel).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
            acc["ra_gtbeta_lfix"].append(((rel_gl - gt_rel).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
            acc["err2d"].append((uv_w - gt2d).norm(dim=-1).mean(-1).cpu().numpy())
            acc["err2d_nounm"].append((uv_w_nounm - gt2d).norm(dim=-1).mean(-1).cpu().numpy())
            acc["gt_reproj_err"].append((project(gt3d, fx, fy, px, py) - gt2d).norm(dim=-1).mean(-1).cpu().numpy())
            acc["reprojD"].append(project(preds["D_pnp_gtbeta_lfix"], fx, fy, px, py).cpu().numpy())
            acc["wilor2d"].append(uv_w.cpu().numpy())
            acc["gt2d"].append(gt2d.cpu().numpy())
            acc["box_center"].append(bc.cpu().numpy())
            acc["box_size"].append(bsz.cpu().numpy())
            acc["idx"].append(batch["idx"].cpu().numpy())
            acc["right"].append(batch["right"].float().cpu().numpy())
            acc["onscreen"].append(onscreen.cpu().numpy())

        if bi % 50 == 0:
            d = np.concatenate(acc["cs_D_pnp_gtbeta"])
            print(f"batch {bi}/{len(loader)}  running CS-MPJPE(D)={d.mean():.1f}mm  "
                  f"gt_reproj_err={np.concatenate(acc['gt_reproj_err']).mean():.2f}px", flush=True)

    R = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(os.path.join(args.out_dir, "per_sample.npz"), imgname=np.array(imgnames), **R)

    subsets = {"all": np.ones(len(R["idx"]), bool), "right": R["right"] > 0.5, "left": R["right"] < 0.5,
               "onscreen": R["onscreen"]}
    gz = R["gt_wrist"][:, 2]
    summary = {"N": {k: int(v.sum()) for k, v in subsets.items()},
               "sanity_gt_reproj_err_px": float(R["gt_reproj_err"].mean()),
               "gt_wrist_z_percentiles_m": {str(p): float(np.percentile(gz, p)) for p in (1, 10, 25, 50, 75, 90, 99)}}
    for sname, msk in subsets.items():
        s = {"wilor2d_err_px": float(R["err2d"][msk].mean()),
             "wilor2d_err_px_nounmirror": float(R["err2d_nounm"][msk].mean()),
             "RA_MPJPE_wilor_beta": float(R["ra_wbeta"][msk].mean()),
             "RA_MPJPE_gt_beta": float(R["ra_gtbeta"][msk].mean()),
             "RA_MPJPE_gt_beta_lfix": float(R["ra_gtbeta_lfix"][msk].mean())}
        for m in METHODS:
            s[m] = method_stats(R, m, msk)
        summary[sname] = s
    depth = {}
    for lo, hi in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
        msk = (gz >= lo) & (gz < hi)
        if msk.sum() == 0:
            continue
        depth[f"{lo:.1f}-{hi:.1f}"] = {m: method_stats(R, m, msk) for m in METHODS}
    summary["depth_bins"] = depth
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 150)
    print(f"N={summary['N']}  sanity: GT3D->K 투영 vs GT2D = {summary['sanity_gt_reproj_err_px']:.3f}px")
    print("GT wrist z percentiles (m): " + "  ".join(f"p{k}={v:.3f}" for k, v in summary["gt_wrist_z_percentiles_m"].items()))
    hdr = (f"{'method':22s} {'CS-MPJPE':>8s} {'med':>6s} | {'wrist bias x/y/z':>22s} | {'wrist MAE x/y/z':>22s} | "
           f"{'|e|':>6s} | {'ray bias':>8s} {'ray MAE':>7s} {'perp':>6s} | {'rp_fit':>6s} {'rp_gt':>6s}")

    def row(name, st):
        w, r = st["wrist"], st["ray"]
        return (f"{name:22s} {st['CS_MPJPE_mean']:8.2f} {st['CS_MPJPE_median']:6.2f} | "
                f"{w['x']['bias']:6.1f} {w['y']['bias']:6.1f} {w['z']['bias']:8.1f} | "
                f"{w['x']['mae']:6.1f} {w['y']['mae']:6.1f} {w['z']['mae']:8.1f} | {w['norm']['mean']:6.1f} | "
                f"{r['bias']:8.1f} {r['mae']:7.1f} {r['perp_mean']:6.1f} | "
                f"{st['rp_fit_px']['mean']:6.2f} {st['rp_gt_px']['mean']:6.2f}")
    for sname in subsets:
        s = summary[sname]
        print(f"\n[{sname}] N={summary['N'][sname]}  WiLoR 2D err={s['wilor2d_err_px']:.2f}px "
              f"(no-unmirror {s['wilor2d_err_px_nounmirror']:.2f}px)  RA-MPJPE wilorβ={s['RA_MPJPE_wilor_beta']:.2f} "
              f"gtβ={s['RA_MPJPE_gt_beta']:.2f} gtβ_lfix={s['RA_MPJPE_gt_beta_lfix']:.2f}mm   (mm / px)")
        print(hdr)
        for m in METHODS:
            print(row(m, s[m]))
    for m in ("D_pnp_gtbeta_lfix", "D_pnp_gtbeta_gn_lfix", "E_pnp_gtbeta_gt2d_lfix", "E_pnp_gtbeta_gt2d_gn_lfix"):
        print(f"\n[depth bins, GT wrist z] {m}")
        print(hdr.replace(f"{'method':22s}", f"{'z bin (m)':22s}"))
        for b, d in depth.items():
            print(row(f"{b} (N={d[m]['N']})", d[m]))

    # 시각화: 랜덤 + D 손목 오차 최악
    def rec(i):
        return dict(imgname=str(imgnames[i]), idx=int(R["idx"][i]), right=float(R["right"][i]),
                    box_center=R["box_center"][i], box_size=float(R["box_size"][i]), gt2d=R["gt2d"][i],
                    wilor2d=R["wilor2d"][i], reprojD=R["reprojD"][i], errD=R["wrist_D_pnp_gtbeta_lfix"][i],
                    err2d=float(R["err2d"][i]))
    rng = np.random.default_rng(0)
    n = len(R["idx"])
    rand_i = rng.choice(n, size=min(args.n_viz_random, n), replace=False)
    worst_i = np.argsort(-np.linalg.norm(R["wrist_D_pnp_gtbeta_lfix"], axis=1))[:args.n_viz_worst]
    save_grid([rec(i) for i in rand_i], os.path.join(args.out_dir, "viz_random.png"))
    save_grid([rec(i) for i in worst_i], os.path.join(args.out_dir, "viz_worst_D.png"))
    print(f"\nsaved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
