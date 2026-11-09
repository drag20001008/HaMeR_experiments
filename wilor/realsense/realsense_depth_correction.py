"""
RealSense D435i 캡처에 depth 보정 파이프라인 적용 (DEPTH_CORRECTION_PIPELINE.md 2절, GT 없음 -> 시각화 목적).

손마다
  1. YOLO 검출 + WiLoR (demo.py와 같은 crop, rescale_factor 2.0)
  2. C 방식: WiLoR beta joint/vertex (왼손 unmirror) + WiLoR 2D + 실제 color K로 translation-only PnP (solve_T)
  3. 손 마스크: SAM3 'hand'/'glove' 후보 중 WiLoR 2D bbox(10% 확장)와 IoU 최대 (IoU <= 0.05면 마스크 없음 -> 보정 안 함)
  4. 예측 mesh를 실제 K로 렌더링(pyrender) -> z_pred, delta = D - z_pred (렌더 & 마스크 & depth 유효 픽셀)
     m = median(delta), c = 사용 픽셀 예측 표면 3D 평균, r = c/|c|, delta_ray = m / r_z, P' = P + delta_ray * r
     필터: 사용 픽셀 < 1000 또는 q90 - q10 > 100mm 이면 보정 안 함
출력: OUT_DIR/<seq>/results.npz (손 단위), OUT_DIR/<seq>/viz/<frame>_h<k>.jpg, OUT_DIR/summary.json
"""
import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import trimesh
from ultralytics import YOLO

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from eval_dexycb_pnp_gtbeta import solve_T

MIN_PIX = 1000
MAX_SPREAD_MM = 100.0
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-8)


class DepthRenderer:
    """OpenCV 카메라 좌표(x 오른쪽, y 아래, z 앞)의 mesh를 실제 K로 렌더링해 픽셀별 표면 z(m)를 돌려줌 (0 = 없음)."""

    def __init__(self, K, W, H):
        self.W, self.H = W, H
        self.cam = pyrender.IntrinsicsCamera(fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"], znear=0.01, zfar=20.0)

    def __call__(self, verts, faces):
        v = verts.copy()
        v[:, 1:] *= -1  # OpenCV -> OpenGL (카메라는 원점, -z를 바라봄)
        scene = pyrender.Scene()
        # 양면 렌더링: DEPTH_ONLY 모드는 material.doubleSided를 무시하고 뒷면을 잘라내므로 면을 반대 방향으로 한 벌 더 넣음.
        # (MANO는 손목이 뚫려 있어 손목 쪽에서 보면 안쪽 면이 보임. extract_depth_anchor.py의 pytorch3d처럼 culling 없이)
        faces2 = np.concatenate([faces, faces[:, [0, 2, 1]]])
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v, faces2, process=False), smooth=False))
        scene.add(self.cam, pose=np.eye(4))
        # demo.py의 Renderer처럼 호출마다 만들고 지움 (만들어 두고 재사용하면 eglMakeCurrent 실패)
        r = pyrender.OffscreenRenderer(self.W, self.H)
        try:
            return r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        finally:
            r.delete()


def project(P, K):
    return np.stack([K["fx"] * P[:, 0] / P[:, 2] + K["cx"], K["fy"] * P[:, 1] / P[:, 2] + K["cy"]], -1)


def backproject(mask, z, K):
    v, u = np.nonzero(mask)
    zz = z[v, u]
    return np.stack([(u - K["cx"]) * zz / K["fx"], (v - K["cy"]) * zz / K["fy"], zz], -1)


def draw_viz(path, img, rec, K, mask, pts_sensor, v_before, v_after, j_before, j_after):
    fig = plt.figure(figsize=(15, 5))

    # (1) 이미지: 마스크 윤곽(노랑) + 보정 전 joint 투영(빨강). 보정은 시선 방향 이동이라 2D 위치는 거의 같음
    ax = fig.add_subplot(1, 3, 1)
    vis = img[:, :, ::-1].copy()
    if mask is not None:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 0), 2)
    uv = project(j_before, K)
    for a, b in BONES:
        cv2.line(vis, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (255, 0, 0), 2)
    ax.imshow(vis)
    ax.set_title(f"{rec['seq']} {rec['frame']} {'R' if rec['right'] else 'L'}  mask={rec['mask_prompt']} IoU {rec['mask_iou']:.2f}",
                 fontsize=9)
    ax.axis("off")

    # (2)(3) 위에서 본 그림(x-z), 옆에서 본 그림(z-y): 센서 depth 점(회색), 보정 전(빨강), 후(초록)
    ref = pts_sensor if len(pts_sensor) else v_before
    c = np.median(ref, 0)
    half = 0.15
    for k, (ia, ib, name) in enumerate(((0, 2, "top view (x-z)"), (2, 1, "side view (z-y)"))):
        ax = fig.add_subplot(1, 3, 2 + k)
        if len(pts_sensor):
            s = pts_sensor[:: max(1, len(pts_sensor) // 4000)]
            ax.scatter(s[:, ia], s[:, ib], s=1, c="0.6", label="sensor depth (mask)")
        ax.scatter(v_before[:, ia], v_before[:, ib], s=1, c="tab:red", alpha=0.5, label="before (PnP)")
        if v_after is not None:
            ax.scatter(v_after[:, ia], v_after[:, ib], s=1, c="tab:green", alpha=0.5, label="after (depth corr.)")
        ax.set_xlim(c[ia] - half, c[ia] + half)
        ax.set_ylim(c[ib] - half, c[ib] + half)
        if ib == 1:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("xyz"[ia] + " (m)")
        ax.set_ylabel("xyz"[ib] + " (m)")
        ax.set_title(name, fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, markerscale=6, loc="upper right")

    status = "corrected" if rec["applied"] else f"NOT corrected ({rec['reason']})"
    fig.suptitle(f"{status} | pixels {rec['n_pix']}  median delta {rec['med_mm']:.1f}mm  q90-q10 {rec['spread_mm']:.1f}mm  "
                 f"shift along ray {rec['shift_mm']:.1f}mm | PnP z {j_before[0, 2] * 1000:.0f}mm", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/nas2/lait/1000_Members/jaehoon/RealSense_D435i_Capture")
    ap.add_argument("--seqs", nargs="+", default=["outdoor_hand_every10", "outdoor_glove_every10",
                                                   "indoor_hand2_every10", "indoor_glove_every10"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_frames", type=int, default=-1)
    args = ap.parse_args()
    out_root = args.out_dir or os.path.join(args.root, "depth_correction")
    dev = torch.device("cuda")

    model, model_cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                  cfg_path="./pretrained_models/model_config.yaml")
    model = model.to(dev).eval()
    detector = YOLO("./pretrained_models/detector.pt").to(dev)
    faces = np.asarray(model.mano.faces, dtype=np.int64)

    summary = {}
    for seq in args.seqs:
        K = json.load(open(os.path.join(args.root, "depth", seq, "intrinsics.json")))["color"]
        W, H = K["width"], K["height"]
        render = DepthRenderer(K, W, H)
        out_dir = os.path.join(out_root, seq)
        os.makedirs(os.path.join(out_dir, "viz"), exist_ok=True)
        frames = sorted(f[:-len("_color.png")] for f in os.listdir(os.path.join(args.root, "images", seq))
                        if f.endswith("_color.png"))
        if args.max_frames > 0:
            frames = frames[:args.max_frames]
        recs, arrays = [], {k: [] for k in ("j_before", "j_after", "v_before", "v_after", "uv_wilor", "betas")}
        n_frames_nodet = 0

        for fr in frames:
            img = cv2.imread(os.path.join(args.root, "images", seq, f"{fr}_color.png"))
            D = cv2.imread(os.path.join(args.root, "depth", seq, f"{fr}_depth_aligned.png"), cv2.IMREAD_UNCHANGED)
            D = D.astype(np.float32) * 0.001  # m
            seg = np.load(os.path.join(args.root, "sam3_seg", seq, f"{fr}.npz"))
            cands = [(p, i, seg[f"{p}_boxes"][i]) for p in ("hand", "glove") for i in range(len(seg[f"{p}_scores"]))]

            det = detector(img, conf=0.3, verbose=False)[0]
            if len(det) == 0:
                n_frames_nodet += 1
                continue
            boxes = np.stack([d.boxes.data.cpu().numpy().squeeze()[:4] for d in det])
            right = np.stack([d.boxes.cls.cpu().numpy().squeeze() for d in det])
            ds = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=2.0)
            batch = recursive_to(torch.utils.data.default_collate([ds[i] for i in range(len(ds))]), dev)
            with torch.no_grad():
                out = model(batch)
            B = len(ds)
            sign = (2 * batch["right"].float() - 1).view(B, 1)

            def unm(x):
                x = x.clone()
                x[..., 0] = x[..., 0] * sign
                return x

            j = out["pred_keypoints_3d"].float()
            v = out["pred_vertices"].float()
            rel_j, rel_v = unm(j - j[:, :1]), unm(v - j[:, :1])
            bc, bsz = batch["box_center"].float(), batch["box_size"].float()
            uv_w = bc[:, None] + unm(out["pred_keypoints_2d"].float()) * bsz.view(B, 1, 1)
            Kt = [torch.full((B,), K[k], device=dev) for k in ("fx", "fy", "cx", "cy")]
            T = solve_T(rel_j, uv_w, *Kt)
            Pj = (rel_j + T[:, None]).cpu().numpy()
            Pv = (rel_v + T[:, None]).cpu().numpy()
            uv_np = uv_w.cpu().numpy()
            betas = out["pred_mano_params"]["betas"].float().cpu().numpy()

            for n in range(B):
                is_right = bool(batch["right"][n].item() > 0.5)
                rec = {"seq": seq, "frame": fr, "hand": n, "right": is_right, "applied": False, "reason": "",
                       "mask_prompt": "none", "mask_idx": -1, "mask_iou": 0.0, "n_pix": 0,
                       "med_mm": float("nan"), "q10_mm": float("nan"), "q90_mm": float("nan"),
                       "spread_mm": float("nan"), "shift_mm": 0.0, "T": T[n].cpu().numpy().tolist()}

                x0, y0 = uv_np[n].min(0)
                x1, y1 = uv_np[n].max(0)
                ex, ey = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
                wbox = [x0 - ex, y0 - ey, x1 + ex, y1 + ey]
                ious = [iou(wbox, b) for _, _, b in cands]
                mask = None
                if ious and max(ious) > 0.05:
                    p, i, _ = cands[int(np.argmax(ious))]
                    mask = seg[f"{p}_masks"][i]
                    rec.update(mask_prompt=p, mask_idx=int(i), mask_iou=float(max(ious)))

                zb = render(Pv[n], faces)
                rend = zb > 0
                # 렌더링 좌표 규약 점검: 렌더 영역 bbox와 정점 투영 bbox 차이(px), 화면 안에 다 들어온 손에서만 의미 있음
                pv = project(Pv[n], K)
                if rend.any():
                    ys, xs = np.nonzero(rend)
                    rec["render_bbox_err_px"] = float(np.abs(np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                                                             - np.r_[pv.min(0), pv.max(0)]).max())
                    rec["fully_onscreen"] = bool((pv.min(0) >= 0).all() and pv[:, 0].max() < W and pv[:, 1].max() < H)
                pts_sensor = np.zeros((0, 3))
                v_after = j_after = None
                if mask is None:
                    rec["reason"] = "no SAM3 mask"
                else:
                    use = rend & mask & (D > 0)
                    pts_sensor = backproject(mask & (D > 0), D, K)
                    rec["n_pix"] = int(use.sum())
                    if rec["n_pix"] > 0:
                        delta = (D[use] - zb[use]) * 1000.0
                        q10, med, q90 = np.percentile(delta, [10, 50, 90])
                        rec.update(med_mm=float(med), q10_mm=float(q10), q90_mm=float(q90), spread_mm=float(q90 - q10))
                    if rec["n_pix"] < MIN_PIX:
                        rec["reason"] = f"pixels < {MIN_PIX}"
                    elif rec["spread_mm"] > MAX_SPREAD_MM:
                        rec["reason"] = f"q90-q10 > {MAX_SPREAD_MM:.0f}mm"
                    else:
                        c = backproject(use, zb, K).mean(0)
                        r = c / np.linalg.norm(c)
                        shift = rec["med_mm"] / 1000.0 / r[2]
                        v_after, j_after = Pv[n] + shift * r, Pj[n] + shift * r
                        rec.update(applied=True, shift_mm=float(shift * 1000.0))

                arrays["j_before"].append(Pj[n])
                arrays["v_before"].append(Pv[n])
                arrays["j_after"].append(j_after if j_after is not None else Pj[n])
                arrays["v_after"].append(v_after if v_after is not None else Pv[n])
                arrays["uv_wilor"].append(uv_np[n])
                arrays["betas"].append(betas[n])
                recs.append(rec)
                draw_viz(os.path.join(out_dir, "viz", f"{fr}_h{n}.jpg"), img, rec, K, mask, pts_sensor,
                         Pv[n], v_after, Pj[n], j_after)
                print(f"{seq}/{fr} h{n} {'R' if is_right else 'L'} mask={rec['mask_prompt']} pix={rec['n_pix']} "
                      f"med={rec['med_mm']:.1f} spread={rec['spread_mm']:.1f} shift={rec['shift_mm']:.1f} "
                      f"{'OK' if rec['applied'] else rec['reason']}", flush=True)

        np.savez_compressed(os.path.join(out_dir, "results.npz"),
                            records=np.array([json.dumps(r) for r in recs]),
                            **{k: np.stack(v) if v else np.zeros((0,)) for k, v in arrays.items()})
        applied = [r for r in recs if r["applied"]]
        shifts = np.array([r["shift_mm"] for r in applied])
        summary[seq] = {
            "frames": len(frames), "frames_no_detection": n_frames_nodet, "hands": len(recs),
            "hands_corrected": len(applied),
            "not_corrected": {k: sum(r["reason"] == k for r in recs) for k in sorted({r["reason"] for r in recs} - {""})},
            "render_bbox_err_px_max_onscreen": max([r["render_bbox_err_px"] for r in recs if r.get("fully_onscreen")], default=None),
            "shift_mm_median": float(np.median(shifts)) if len(shifts) else None,
            "shift_mm_q10_q90": np.percentile(shifts, [10, 90]).tolist() if len(shifts) else None,
        }
        print(json.dumps({seq: summary[seq]}, indent=1), flush=True)

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DEPTH_CORRECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
