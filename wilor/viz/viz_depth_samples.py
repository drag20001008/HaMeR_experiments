"""D_lfix / GT depth + 실제 손 마스크((b) median 보정) / GT 를 실제 샘플 30개(onscreen 무작위)로 비교.
샘플마다 [이미지 crop + 2D 재투영] [위에서 본 X-Z (GT 손목 원점, mm)]. 샘플당 파일 하나.
"""
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/nas2/lait/1000_Members/jaehoon/DexYCB"
OUT = "depth_anchor_extract_gtdepth"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
C_D, C_B = "#2a78d6", "#eb6834"  # palette slots 1, 2
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
N_SAMPLES, PER_PAGE, SEED = 30, 10, 1

zs = [np.load(f) for f in sorted(glob.glob(f"{OUT}/shard*.npz"))]
cat = lambda k: np.concatenate([z[k] for z in zs])
idx, on = cat("idx"), cat("onscreen").astype(bool)
P, G = cat("pred_j"), cat("gt_j")
med, cnt, cen = cat("pred_med")[:, 0, 0], cat("pred_cnt")[:, 0, 0], cat("pred_cen")[:, 0, 0]
r = cen / np.linalg.norm(cen, axis=-1, keepdims=True).clip(1e-9)
use = (cnt >= 1000) & np.isfinite(med)
P_b = P + (np.where(use, med / 1000 / r[:, 2].clip(1e-6), 0)[:, None] * r)[:, None, :]

ann = np.load(f"{ROOT}/wilor_annotations_v3.npz")
K = {k: ann[k][idx] for k in ("fx", "fy", "px", "py")}
imgname, center, bsz, right = ann["imgname"][idx], ann["center"][idx], cat("box_size"), cat("right")
pick = np.random.default_rng(SEED).choice(np.where(on)[0], N_SAMPLES, replace=False)


def proj(X, i):
    return np.stack([K["fx"][i] * X[:, 0] / X[:, 2] + K["px"][i], K["fy"][i] * X[:, 1] / X[:, 2] + K["py"][i]], -1)


def skel(ax, pts, color, ls, lw):
    for u, v in BONES:
        ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], color=color, ls=ls, lw=lw, solid_capstyle="round")
    ax.plot(pts[0, 0], pts[0, 1], "o", color=color, ms=4.5)


os.makedirs(f"{OUT}/depth_samples", exist_ok=True)
handles = [plt.Line2D([], [], color=INK, lw=2, label="GT"),
           plt.Line2D([], [], color=C_D, lw=1.6, label="D_lfix (no depth)"),
           plt.Line2D([], [], color=C_B, lw=1.6, ls="-", label="GT depth + hand mask (median)")]
for n, i in enumerate(pick, 1):
    fig, (ax_img, ax_top) = plt.subplots(1, 2, figsize=(10, 4.9), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.15]))
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, str(imgname[i]))), cv2.COLOR_BGR2RGB)
    cx, cy = center[i]
    half = 0.65 * bsz[i]
    ax_img.imshow(img)
    ax_img.set_xlim(cx - half, cx + half)
    ax_img.set_ylim(cy + half, cy - half)
    ax_img.axis("off")
    skel(ax_img, proj(G[i], i), INK, "-", 2.0)
    skel(ax_img, proj(P[i], i), C_D, "-", 1.2)
    skel(ax_img, proj(P_b[i], i), C_B, "-", 1.2)
    dz_d, dz_b = (P[i][0, 2] - G[i][0, 2]) * 1000, (P_b[i][0, 2] - G[i][0, 2]) * 1000
    fig.suptitle(f"#{int(idx[i])} ({'right' if right[i] > .5 else 'left'} hand)   wrist Δz vs GT:  D_lfix {dz_d:+.0f} mm  |  GT depth + hand mask {dz_b:+.0f} mm",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    ax_img.set_title("image + 2D reprojection", color=INK2, fontsize=9, loc="left")

    o = G[i][0]
    rel = lambda X: ((X - o) * 1000)[:, [0, 2]]  # X(right), Z(depth)
    ax_top.set_facecolor(SURF)
    for s_ in ("top", "right"):
        ax_top.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax_top.spines[s_].set_color(INK2)
    ax_top.tick_params(colors=INK2, labelsize=8)
    ax_top.grid(True, color=GRID, lw=0.8)
    ax_top.set_axisbelow(True)
    skel(ax_top, rel(G[i]), INK, "-", 2.0)
    skel(ax_top, rel(P[i]), C_D, "-", 1.4)
    skel(ax_top, rel(P_b[i]), C_B, "-", 1.4)
    allp = np.concatenate([rel(G[i]), rel(P[i]), rel(P_b[i])])
    lo, hi = allp.min(0), allp.max(0)
    c, s = (lo + hi) / 2, max((hi - lo).max() / 2 + 8, 60)
    ax_top.set_xlim(c[0] - s, c[0] + s)
    ax_top.set_ylim(c[1] - s, c[1] + s)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (mm, right →)", color=INK2, fontsize=8.5)
    ax_top.set_ylabel("Z depth (mm, farther ↑)", color=INK2, fontsize=8.5)
    ax_top.set_title("top view (X–Z), origin = GT wrist", color=INK2, fontsize=9, loc="left")
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = f"{OUT}/depth_samples/sample{n:02d}_idx{int(idx[i])}.png"
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)
print(f"saved {len(pick)} files to {OUT}/depth_samples/")
dz = (P[pick][:, 0, 2] - G[pick][:, 0, 2]) * 1000
dzb = (P_b[pick][:, 0, 2] - G[pick][:, 0, 2]) * 1000
print(f"30 samples wrist |Δz| mean: D_lfix {np.abs(dz).mean():.1f}  GT depth {np.abs(dzb).mean():.1f} mm")
