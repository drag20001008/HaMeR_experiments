"""D_lfix / GT depth + 실제 손 마스크 (b 보정) / oracle δ* 를 GT와 비교 시각화 (onscreen).
  three_methods_dist.png   : 손목 x/y/z 부호 있는 오차 분포 + CS-MPJPE 누적분포
  three_methods_samples.png: 샘플별 [이미지 + 2D 재투영] [위에서 본 X-Z] [옆에서 본 Y-Z]
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
M_COL = {"D_lfix": "#2a78d6", "GT depth + hand mask": "#eb6834", "oracle δ*": "#1baf7a"}  # palette slots 1-3
M_LS = {"D_lfix": "-", "GT depth + hand mask": "--", "oracle δ*": ":"}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def load(d):
    zs = [np.load(f) for f in sorted(glob.glob(f"{d}/shard*.npz"))]
    return lambda k: np.concatenate([z[k] for z in zs])


Gd = load(OUT)
idx, on = Gd("idx"), Gd("onscreen").astype(bool)
P, G = Gd("pred_j"), Gd("gt_j")
# (b) 보정: GT depth, 실제 손 마스크, erode 0, all, median, >=1000px, 필터 없음
med, cnt, cen = Gd("pred_med")[:, 0, 0], Gd("pred_cnt")[:, 0, 0], Gd("pred_cen")[:, 0, 0]
r = cen / np.linalg.norm(cen, axis=-1, keepdims=True).clip(1e-9)
use = (cnt >= 1000) & np.isfinite(med)
P_gd = P + (np.where(use, med / 1000 / r[:, 2].clip(1e-6), 0)[:, None] * r)[:, None, :]
cj = P.mean(1)
rj = cj / np.linalg.norm(cj, axis=-1, keepdims=True)
P_or = P + (((G[..., 2].mean(1) - P[..., 2].mean(1)) / rj[:, 2])[:, None] * rj)[:, None, :]
METH = {"D_lfix": P, "GT depth + hand mask": P_gd, "oracle δ*": P_or}

ann = np.load(f"{ROOT}/wilor_annotations_v3.npz")
K = {k: ann[k][idx] for k in ("fx", "fy", "px", "py")}
imgname, center, bsz = ann["imgname"][idx], ann["center"][idx], Gd("box_size")


def style(ax, grid_axis="both"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------- 1. distributions ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURF)
fig.suptitle(f"DexYCB s0 test, onscreen (N={on.sum():,}): wrist error vs GT, pred − GT", color=INK, fontsize=12.5,
             x=0.02, ha="left")
bins = np.arange(-60, 61, 1)
ctr = (bins[:-1] + bins[1:]) / 2
for ai, (ax, axis) in enumerate(zip(axes.flat[:3], "xyz")):
    style(ax)
    for m, Pk in METH.items():
        e = (Pk[on][:, 0, ai] - G[on][:, 0, ai]) * 1000
        h = np.histogram(e, bins)[0] / on.sum() * 100
        ax.plot(ctr, h, color=M_COL[m], ls=M_LS[m], lw=2,
                label=f"{m}  (bias {e.mean():+.1f}, MAE {np.abs(e).mean():.1f})")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-60, 60)
    ax.set_title(f"wrist {axis} error (mm)" + ("   ← nearer camera | farther →" if axis == "z" else ""),
                 color=INK, loc="left", fontsize=10.5)
    ax.set_ylabel("% per 1 mm", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
ax = axes.flat[3]
style(ax)
for m, Pk in METH.items():
    cs = np.sort(np.linalg.norm(Pk[on] - G[on], axis=-1).mean(-1) * 1000)
    ax.plot(cs, np.arange(1, len(cs) + 1) / len(cs) * 100, color=M_COL[m], ls=M_LS[m], lw=2,
            label=f"{m}  (mean {cs.mean():.1f}, median {np.median(cs):.1f})")
ax.set_xlim(0, 60)
ax.set_ylim(0, 100)
ax.set_title("CS-MPJPE, cumulative (% of samples ≤ x mm)", color=INK, loc="left", fontsize=10.5)
ax.set_xlabel("CS-MPJPE (mm)", color=INK2)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUT}/three_methods_dist.png", dpi=130, facecolor=SURF)
plt.close(fig)

# ---------------- 2. samples ----------------
rng = np.random.default_rng(0)
cand = np.where(on & use)[0]
ez_d = (P[:, 0, 2] - G[:, 0, 2]) * 1000
pick = list(rng.choice(cand, 4, replace=False)) + list(cand[np.argsort(-np.abs(ez_d[cand]))[:4]])
tags = ["random"] * 4 + ["large D_lfix z error"] * 4


def proj(X, i):
    return np.stack([K["fx"][i] * X[:, 0] / X[:, 2] + K["px"][i], K["fy"][i] * X[:, 1] / X[:, 2] + K["py"][i]], -1)


def skel(ax, pts, color, ls, lw, a, b):
    for u, v in BONES:
        ax.plot([pts[u, a], pts[v, a]], [pts[u, b], pts[v, b]], color=color, ls=ls, lw=lw)
    ax.plot(pts[0, a], pts[0, b], "o", color=color, ms=5)


fig, axes = plt.subplots(len(pick), 3, figsize=(13, 3.3 * len(pick)), facecolor=SURF,
                         gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
fig.suptitle("GT (black) vs D_lfix / GT depth + hand mask / oracle δ*   —  views centered on GT wrist (mm)",
             color=INK, fontsize=12, x=0.02, ha="left")
for row, (i, tag) in enumerate(zip(pick, tags)):
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, str(imgname[i]))), cv2.COLOR_BGR2RGB)
    cx, cy = center[i]
    half = 0.65 * bsz[i]
    ax = axes[row, 0]
    ax.imshow(img)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)
    ax.axis("off")
    skel(ax, proj(G[i], i), INK, "-", 2.2, 0, 1)
    for m, Pk in METH.items():
        skel(ax, proj(Pk[i], i), M_COL[m], M_LS[m], 1.3, 0, 1)
    short = {"D_lfix": "D_lfix", "GT depth + hand mask": "GTdepth", "oracle δ*": "oracle"}
    errs = "  ".join(f"{short[m]} Δz {(Pk[i][0, 2] - G[i][0, 2]) * 1000:+.0f}" for m, Pk in METH.items())
    ax.set_title(f"#{int(idx[i])} ({tag})\n{errs} mm", color=INK, fontsize=8.5, loc="left")
    o = G[i][0]
    for col, (a, b, xl, yl) in enumerate(((0, 2, "X (right)", "Z (depth, farther ↑)"), (1, 2, "Y (down)", "Z (depth, farther ↑)")), 1):
        ax = axes[row, col]
        style(ax)
        rel = lambda X: (X - o) * 1000
        skel(ax, rel(G[i]), INK, "-", 2.2, a, b)
        for m, Pk in METH.items():
            skel(ax, rel(Pk[i]), M_COL[m], M_LS[m], 1.5, a, b)
        allp = np.concatenate([rel(G[i])] + [rel(Pk[i]) for Pk in METH.values()])
        lo, hi = allp[:, [a, b]].min(0), allp[:, [a, b]].max(0)
        c, s = (lo + hi) / 2, max((hi - lo).max() / 2 + 10, 60)
        ax.set_xlim(c[0] - s, c[0] + s)
        ax.set_ylim(c[1] - s, c[1] + s)
        ax.set_aspect("equal")
        ax.set_xlabel(xl, color=INK2, fontsize=8)
        ax.set_ylabel(yl, color=INK2, fontsize=8)
        if row == 0:
            ax.set_title("top view (X–Z)" if col == 1 else "side view (Y–Z)", color=INK, fontsize=10, loc="left")
handles = [plt.Line2D([], [], color=INK, lw=2.2, label="GT")] + \
          [plt.Line2D([], [], color=M_COL[m], ls=M_LS[m], lw=1.8, label=m) for m in METH]
fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(f"{OUT}/three_methods_samples.png", dpi=110, facecolor=SURF)
print("saved", f"{OUT}/three_methods_dist.png", f"{OUT}/three_methods_samples.png")
