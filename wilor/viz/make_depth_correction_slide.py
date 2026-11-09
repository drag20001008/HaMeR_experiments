"""발표용 방법 설명 그림: depth 방향 보정 (b) 방식, GT depth(GT mesh 렌더링) 예시 1샘플. hamer env (pytorch3d).
 ① Δ(p) = D(p) − z_pred(p)  (예측 손 렌더링 ∩ 손 마스크 ∩ depth 유효 픽셀)
 ② m = median Δ
 ③ 손 중심 c를 지나는 시선 방향 r̂ = c/‖c‖ 로 δ = m / r̂_z 만큼 강체 평행이동: P' = P + δ r̂
"""
import copy
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import smplx
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from torch.utils.data import DataLoader

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset
from eval_dexycb_pnp_gtbeta import solve_T
from extract_depth_anchor import ROOT, TIPS, REORDER, WithDepth

ROW = 390800
OUT = "depth_anchor_extract_gtdepth/slide_depth_correction.png"
H, W = 480, 640
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
C_BEFORE, C_AFTER, C_RAY = "#2a78d6", "#eb6834", "#8f8e87"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def main():
    dev = torch.device("cuda")
    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt", cfg_path="./pretrained_models/model_config.yaml")
    model = model.to(dev).eval()
    mano_x = copy.deepcopy(model.mano)
    mano_x.shapedirs[:, 0, :] *= -1
    gtL = {s: smplx.MANO(model_path="mano_data_dexycb", is_rhand=(s == "right"), use_pca=False, num_pca_comps=45,
                         flat_hand_mean=False).to(dev) for s in ("right", "left")}
    comps = {s: torch.tensor(np.asarray(L.np_hand_components), dtype=torch.float32, device=dev) for s, L in gtL.items()}
    ds = ImageDataset(cfg, f"{ROOT}/wilor_annotations_v3.npz", ROOT, train=False, rescale_factor=-1)
    batch = next(iter(DataLoader(WithDepth(ds, np.array([ROW])), batch_size=1, num_workers=0)))
    to = lambda x: x.to(dev) if isinstance(x, torch.Tensor) else ({k: to(v) for k, v in x.items()} if isinstance(x, dict) else x)
    batch = to(batch)
    with torch.no_grad():
        out = model.forward_step(batch, train=False)
        right = batch["right"].float()
        is_left = bool(right[0] < 0.5)
        sign = (2 * right - 1).view(1, 1)

        def unm(x):
            x = x.clone()
            x[..., 0] = x[..., 0] * sign
            return x

        fx, fy, px, py = [batch[k].float() for k in ("fx", "fy", "px", "py")]
        betas = batch["mano_params"]["betas"].float()
        pm = out["pred_mano_params"]
        go, hp = pm["global_orient"].reshape(1, -1, 3, 3), pm["hand_pose"].reshape(1, -1, 3, 3)
        o = (mano_x if is_left else model.mano)(global_orient=go, hand_pose=hp, betas=betas, pose2rot=False)
        rel_j, rel_v = unm(o.joints - o.joints[:, :1]).float(), unm(o.vertices - o.joints[:, :1]).float()
        uv = batch["box_center"].float()[:, None] + unm(out["pred_keypoints_2d"].float()) * batch["box_size"].float().view(1, 1, 1)
        T = solve_T(rel_j, uv, fx, fy, px, py)
        Pj, Pv = rel_j + T[:, None], rel_v + T[:, None]
        side = "left" if is_left else "right"
        pmg = batch["pose_m"].float()
        og = gtL[side](global_orient=pmg[:, :3], hand_pose=pmg[:, 3:48] @ comps[side], betas=betas, transl=pmg[:, 48:51])
        Gv = og.vertices.float()
        Gj = torch.cat([og.joints[:, :16], og.vertices[:, TIPS[side]]], 1)[:, REORDER].float()

        Km = torch.zeros(1, 3, 3, device=dev)
        Km[:, 0, 0], Km[:, 1, 1], Km[:, 0, 2], Km[:, 1, 2], Km[:, 2, 2] = fx, fy, px, py, 1
        cams = cameras_from_opencv_projection(torch.eye(3, device=dev)[None], torch.zeros(1, 3, device=dev), Km,
                                              torch.tensor([[H, W]], device=dev))
        raster = MeshRasterizer(raster_settings=RasterizationSettings(image_size=(H, W), faces_per_pixel=1, bin_size=0))
        f_pred = torch.as_tensor(np.asarray(model.mano.faces, dtype=np.int64), device=dev)
        f_gt = torch.as_tensor(np.asarray(gtL[side].faces, dtype=np.int64), device=dev)
        zp = raster(Meshes(verts=[Pv[0]], faces=[f_pred]), cameras=cams).zbuf[0, ..., 0].cpu().numpy()
        zg = raster(Meshes(verts=[Gv[0]], faces=[f_gt]), cameras=cams).zbuf[0, ..., 0].cpu().numpy()
    K = [float(t[0]) for t in (fx, fy, px, py)]
    Pj, Gj = Pj[0].cpu().numpy(), Gj[0].cpu().numpy()
    hand = batch["seg"][0].cpu().numpy() == 255
    D = np.where(zg > 0, zg, 0)  # GT depth (rendered GT mesh)
    used = (zp > 0) & hand & (D > 0)
    delta = np.where(used, (D - zp) * 1000, np.nan)
    m = float(np.nanmedian(delta))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([(uu - K[2]) * zp / K[0], (vv - K[3]) * zp / K[1], zp], -1)[used]
    c = pts.mean(0)
    r = c / np.linalg.norm(c)
    dlt = m / 1000 / r[2]
    Pj2 = Pj + dlt * r
    print(f"#{ROW} {side}: used px {used.sum()}, m {m:+.1f} mm, r_z {r[2]:.3f}, delta {dlt * 1000:+.1f} mm, "
          f"wrist dz before {(Pj[0, 2] - Gj[0, 2]) * 1000:+.1f} after {(Pj2[0, 2] - Gj[0, 2]) * 1000:+.1f} mm")

    # ---------------- figure (16:9) ----------------
    plt.rcParams.update({"font.size": 13})
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.935, "Depth correction: shift the hand along the viewing ray by the median surface-depth offset",
             fontsize=19, color=INK, weight="bold")
    TY = 0.85  # 세 칸 제목 높이 통일
    for x, t in ((0.03, "① Per-pixel depth difference"), (0.375, "② Robust depth offset"),
                 (0.695, "③ Shift along the viewing ray (top view)")):
        fig.text(x, TY, t, fontsize=15, color=INK, weight="bold")

    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, str(np.load(f"{ROOT}/wilor_annotations_v3.npz")["imgname"][ROW]))),
                       cv2.COLOR_BGR2RGB)
    ys, xs = np.where(used)
    cx, cy = xs.mean(), ys.mean()
    half = 0.75 * max(xs.max() - xs.min(), ys.max() - ys.min())

    def style(ax):
        ax.set_facecolor(SURF)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=10)

    # ① per-pixel difference map (발산형: 보라=가까움, 회색=0, 노랑=멂; ③의 before/after 색과 겹치지 않게)
    ax = fig.add_axes([0.03, 0.30, 0.27, 0.52])
    ax.imshow(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB), alpha=0.55)
    cmap = LinearSegmentedColormap.from_list("div", ["#4a3aa7", "#e4e3dd", "#eda100"])
    vmax = np.nanpercentile(np.abs(delta), 98)
    im = ax.imshow(delta, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.axis("off")
    cax = fig.add_axes([0.305, 0.33, 0.009, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors=INK2, labelsize=10)
    cb.set_label("Δ(p)  (mm)", fontsize=10.5, color=INK2)
    fig.text(0.03, 0.205, r"$\Delta(p) = D(p) - z_{\mathrm{pred}}(p)$", fontsize=17, color=INK)
    fig.text(0.03, 0.155, "D: depth map (here: GT depth)\n"
                         r"$z_{\mathrm{pred}}$: rendered depth of the predicted hand" "\n"
                         "p: pixels in predicted hand ∩ hand mask ∩ valid depth\n"
                         "purple: actual hand nearer · yellow: farther",
             fontsize=11.5, color=INK2, linespacing=1.55, va="top")

    # ② median
    ax = fig.add_axes([0.40, 0.33, 0.26, 0.49])
    style(ax)
    v = delta[used]
    bins = np.linspace(np.percentile(v, 0.5) - 3, np.percentile(v, 99.5) + 3, 45)
    n_, _, _ = ax.hist(v, bins=bins, color=C_RAY, edgecolor=SURF, linewidth=1.5)
    ax.axvline(m, color=INK, lw=2.5)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.annotate(f"m = median Δ\n= {m:+.1f} mm", xy=(m, n_.max() * 0.97), xytext=(-14, 0), textcoords="offset points",
                fontsize=14, color=INK, va="top", ha="right")
    ax.set_xlabel("Δ(p)  (mm)     ← actual hand nearer | farther →", color=INK2)
    ax.set_ylabel("# pixels", color=INK2)
    fig.text(0.375, 0.205, r"$m = \mathrm{median}_{p}\ \Delta(p)$", fontsize=17, color=INK)
    fig.text(0.375, 0.155, "median: robust to boundary / background pixels\n"
                          f"m < 0 → predicted hand is too far (here {m:+.0f} mm)",
             fontsize=11.5, color=INK2, linespacing=1.55, va="top")

    # ③ shift along ray (top view; camera distance not to scale)
    ax = fig.add_axes([0.71, 0.30, 0.28, 0.52])
    style(ax)
    o = c
    top = lambda X: ((X - o) * 1000)[..., [0, 2]]
    rr = r[[0, 2]] / np.linalg.norm(r[[0, 2]])
    cam = -170.0 * rr
    ray_end = 90 * rr
    ax.plot([cam[0], ray_end[0]], [cam[1], ray_end[1]], color=C_RAY, lw=1.5, ls=(0, (5, 4)), zorder=1)
    ax.plot(*cam, marker="^", ms=16, color=INK2, zorder=5)
    ax.text(cam[0] + 10, cam[1], "camera (not to scale)", fontsize=10.5, color=INK2, va="center")

    def sk(J, col, lw):
        t = top(J)
        for a, b in BONES:
            ax.plot([t[a, 0], t[b, 0]], [t[a, 1], t[b, 1]], color=col, lw=lw, solid_capstyle="round", zorder=3)
    sk(Gj, INK, 2.2)
    sk(Pj, C_BEFORE, 1.8)
    sk(Pj2, C_AFTER, 1.8)
    c2 = top(c + dlt * r)
    ax.annotate("", xy=c2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4, mutation_scale=20), zorder=7)
    ax.plot(0, 0, "o", color=C_BEFORE, ms=9, zorder=8, mec=SURF, mew=1.5)
    ax.plot(*c2, "o", color=C_AFTER, ms=9, zorder=8, mec=SURF, mew=1.5)
    allp = np.concatenate([top(Gj), top(Pj), top(Pj2), cam[None], [[0, 0]]])
    lo, hi = allp.min(0) - 25, allp.max(0) + 25
    ctr, s_ = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - s_, ctr[0] + s_ + 40); ax.set_ylim(ctr[1] - s_, ctr[1] + s_)
    ax.set_aspect("equal")
    ax.annotate("c", xy=(0, 0), xytext=(ctr[0] + s_ - 5, 18), fontsize=14, color=C_BEFORE, weight="bold",
                arrowprops=dict(arrowstyle="-", color=C_BEFORE, lw=0.8), va="center")
    ax.annotate(r"$c + \delta\,\hat r$", xy=c2, xytext=(ctr[0] + s_ - 5, c2[1] - 20), fontsize=14, color=C_AFTER,
                arrowprops=dict(arrowstyle="-", color=C_AFTER, lw=0.8), va="center")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("X (mm)", color=INK2); ax.set_ylabel("depth Z (mm)   farther ↑", color=INK2)
    ax.legend(handles=[plt.Line2D([], [], color=INK, lw=2.2, label="GT"),
                       plt.Line2D([], [], color=C_BEFORE, lw=1.8, label="before (PnP only)"),
                       plt.Line2D([], [], color=C_AFTER, lw=1.8, label="after correction")],
              loc="lower left", frameon=False, fontsize=11, labelcolor=INK)
    fig.text(0.695, 0.205, r"$\hat r = c/\|c\|,\quad \delta = m/\hat r_z,\quad P' = P + \delta\,\hat r$", fontsize=16, color=INK)
    fig.text(0.695, 0.155, "c: 3D centroid of the used predicted-surface pixels\n"
                          "rigid shift: hand shape & rotation unchanged\n"
                          "c stays on its viewing ray → its 2D position is unchanged",
             fontsize=11.5, color=INK2, linespacing=1.55, va="top")
    fig.savefig(OUT, dpi=150, facecolor=SURF)
    print("saved", OUT)
    draw_topview(Gj, Pj, Pj2, c, r, dlt, m)
    draw_topview(Gj, Pj, Pj2, c, r, dlt, m, out="depth_anchor_extract_gtdepth/depth_correction_topview_clean.png", clean=True)


def draw_topview(Gj, Pj, Pj2, c, r, dlt, m, out="depth_anchor_extract_gtdepth/depth_correction_topview.png", clean=False):
    """③번 칸만 단독 그림으로: 탑뷰에서 시선 방향 보정."""
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURF)
    ax.set_facecolor(SURF)
    top = lambda X: ((X - c) * 1000)[..., [0, 2]]
    rr = r[[0, 2]] / np.linalg.norm(r[[0, 2]])
    cam = -115.0 * rr
    ax.plot([cam[0], 95 * rr[0]], [cam[1], 95 * rr[1]], color=C_RAY, lw=1.4, ls=(0, (6, 4)), zorder=1)
    ax.plot(*cam, marker="^", ms=18, color=INK2, zorder=5)
    if not clean:
        ax.text(cam[0] + 12, cam[1], "camera\n(distance not to scale)", fontsize=10.5, color=INK2, va="center")

    def sk(J, col, lw, z):
        t = top(J)
        for a_, b_ in BONES:
            ax.plot([t[a_, 0], t[b_, 0]], [t[a_, 1], t[b_, 1]], color=col, lw=lw, solid_capstyle="round", zorder=z)
        ax.plot(t[1:, 0], t[1:, 1], "o", color=col, ms=2.6, zorder=z)
        ax.plot(t[0, 0], t[0, 1], "o", color=col, ms=6, zorder=z, mec=SURF, mew=1)
    sk(Pj, C_BEFORE, 2.0, 3)
    sk(Pj2, C_AFTER, 2.0, 3)
    sk(Gj, INK, 1.6, 4)
    c2 = top(c + dlt * r)
    ax.annotate("", xy=c2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6, mutation_scale=22), zorder=7)
    ax.plot(0, 0, "o", color=C_BEFORE, ms=10, zorder=8, mec=INK, mew=1.3)
    ax.plot(*c2, "o", color=C_AFTER, ms=10, zorder=8, mec=INK, mew=1.3)

    allp = np.concatenate([top(Gj), top(Pj), top(Pj2), cam[None]])
    lo, hi = allp.min(0) - 22, allp.max(0) + 22
    ctr, s_ = (lo + hi) / 2, (hi - lo).max() / 2
    hand_right = np.concatenate([top(Gj), top(Pj), top(Pj2)])[:, 0].max()
    ax.set_xlim(ctr[0] - s_, ctr[0] + s_ + (0 if clean else 95)); ax.set_ylim(ctr[1] - s_, ctr[1] + s_)
    ax.set_aspect("equal")
    xr = hand_right + 18  # 라벨 열: 손 오른쪽 바깥
    if not clean:
        ax.annotate("c: centroid of the used\npredicted-surface pixels", xy=(0, 0), xytext=(xr, 12), fontsize=11, color=C_BEFORE,
                    va="center", arrowprops=dict(arrowstyle="-", color=C_BEFORE, lw=0.8, shrinkB=6))
        ax.annotate(r"$c + \delta\,\hat r$", xy=c2, xytext=(xr, c2[1] - 22), fontsize=13, color=C_AFTER,
                    va="center", arrowprops=dict(arrowstyle="-", color=C_AFTER, lw=0.8, shrinkB=6))
        mid = c2 / 2
        ax.annotate(rf"$\delta\,\hat r$  ($\delta$ = {dlt * 1000:+.0f} mm)", xy=mid, xytext=(xr, mid[1] - 4), fontsize=12,
                    color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=4))
        ax.text(0.47, 0.27,
                rf"$m = \mathrm{{median}}_p\,[D(p) - z_{{\mathrm{{pred}}}}(p)] = {m:+.0f}$ mm" "\n"
                r"$\hat r = c/\|c\|,\quad \delta = m/\hat r_z,\quad P' = P + \delta\,\hat r$",
                fontsize=12, color=INK, va="top", linespacing=1.8, transform=ax.transAxes)
    for s2 in ("top", "right"):
        ax.spines[s2].set_visible(False)
    for s2 in ("left", "bottom"):
        ax.spines[s2].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("X (mm)", color=INK2)
    ax.set_ylabel("depth Z (mm),  farther ↑", color=INK2)
    ax.legend(handles=[plt.Line2D([], [], color=INK, lw=1.8, label="GT"),
                       plt.Line2D([], [], color=C_BEFORE, lw=2, label="before (PnP only)"),
                       plt.Line2D([], [], color=C_AFTER, lw=2, label="after depth correction")],
              loc="lower left", frameon=False, fontsize=11, labelcolor=INK)
    if not clean:
        ax.set_title("Shifting the hand along the viewing ray (top view)", loc="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURF)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
