"""
per_sample.npz에서 손목 depth(z) 오차 분포를 그린다 (onscreen 샘플).
  (1) signed z 오차 히스토그램   (2) |z 오차| 누적분포   (3) GT depth 구간별 z 오차 중앙값 + IQR
A_demo_fakef는 z 오차가 약 16m라 같은 축에 못 올려서 수치만 출력한다.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = [  # (key, label, color: reference palette slot 1-4 in fixed order, linestyle)
    ("B_cam_realK", "B: WiLoR cam + real K", "#2a78d6", "-"),
    ("C_pnp_wbeta", "C: WiLoR β + PnP", "#eb6834", "--"),
    ("D_pnp_gtbeta_lfix", "D_lfix: GT β + PnP", "#1baf7a", "-"),
    ("E_pnp_gtbeta_gt2d_lfix", "E_lfix: GT β + GT 2D (oracle)", "#eda100", ":"),
]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    on = d["onscreen"]
    gz = d["gt_wrist"][:, 2]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.npz)), "depth_error.png")

    ez = {k: d[f"wrist_{k}"][:, 2] for k, *_ in SERIES}
    ezA = d["wrist_A_demo_fakef"][:, 2]

    print(f"onscreen N={on.sum()}  (A_demo_fakef z err median {np.median(ezA[on]) / 1000:.2f} m)")
    print(f"{'method':26s} {'bias':>6s} {'med':>6s} {'IQR':>13s} | {'|e|p50':>6s} {'p75':>6s} {'p90':>6s} {'p95':>6s} | "
          f"{'<10mm':>6s} {'<20mm':>6s} {'<50mm':>6s} | {'rel|e| med':>9s}")
    for k, lab, *_ in SERIES:
        e = ez[k][on]
        a = np.abs(e)
        q1, q3 = np.percentile(e, [25, 75])
        rel = np.median(a / (gz[on] * 1000)) * 100
        print(f"{k:26s} {e.mean():6.1f} {np.median(e):6.1f} [{q1:5.1f},{q3:5.1f}] | "
              f"{np.percentile(a, 50):6.1f} {np.percentile(a, 75):6.1f} {np.percentile(a, 90):6.1f} {np.percentile(a, 95):6.1f} | "
              f"{(a < 10).mean() * 100:5.1f}% {(a < 20).mean() * 100:5.1f}% {(a < 50).mean() * 100:5.1f}% | {rel:8.2f}%")

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), facecolor=SURF)
    fig.suptitle(f"DexYCB s0 test, onscreen (N={on.sum():,}): wrist depth (z) error, pred − GT",
                 color=INK, fontsize=13, x=0.02, ha="left")

    # (1) signed histogram
    ax = axes[0]
    style(ax)
    bins = np.arange(-150, 152, 2)
    for k, lab, c, ls in SERIES:
        e = ez[k][on]
        h, _ = np.histogram(e, bins=bins)  # 범위 밖은 제외 (끝 칸에 몰지 않음)
        ctr = (bins[:-1] + bins[1:]) / 2
        out_pct = (np.abs(e) > 150).mean() * 100
        ax.plot(ctr, h / on.sum() * 100, color=c, ls=ls, lw=2,
                label=f"{lab}  (median {np.median(e):+.1f}, |e|>150: {out_pct:.1f}%)")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-150, 150)
    ax.set_xlabel("signed z error (mm)   ← too near  |  too far →   (samples beyond ±150 not drawn)", color=INK2)
    ax.set_ylabel("% of samples per 2 mm bin", color=INK2)
    ax.set_title("Distribution of signed depth error", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    # (2) CDF of |e_z|
    ax = axes[1]
    style(ax)
    for k, lab, c, ls in SERIES:
        a = np.sort(np.abs(ez[k][on]))
        y = np.arange(1, len(a) + 1) / len(a) * 100
        ax.plot(a, y, color=c, ls=ls, lw=2,
                label=f"{lab}  (p50 {np.percentile(a, 50):.1f}, p90 {np.percentile(a, 90):.1f} mm)")
    for t in (10, 20, 50):
        ax.axvline(t, color=GRID, lw=1.5, zorder=0)
    ax.axhline(50, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.set_xlabel("|z error| (mm)", color=INK2)
    ax.set_ylabel("% of samples with error ≤ x", color=INK2)
    ax.set_title("Cumulative distribution of absolute depth error", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")

    # (3) by GT depth: median + IQR
    ax = axes[2]
    style(ax)
    edges = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.30])
    ctr = (edges[:-1] + edges[1:]) / 2
    offs = np.linspace(-0.018, 0.018, len(SERIES))
    for (k, lab, c, ls), o in zip(SERIES, offs):
        med, lo, hi = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = on & (gz >= a) & (gz < b)
            e = ez[k][m]
            med.append(np.median(e)); lo.append(np.percentile(e, 25)); hi.append(np.percentile(e, 75))
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.errorbar(ctr + o, med, yerr=[med - lo, hi - med], color=c, ls=ls, lw=2, marker="o", ms=5,
                    capsize=3, label=lab)
    counts = [int((on & (gz >= a) & (gz < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(ctr)
    ax.set_xticklabels([f"{a:.2f}–{b:.2f}\nN={n:,}" for a, b, n in zip(edges[:-1], edges[1:], counts)], fontsize=8)
    ax.set_xlabel("GT wrist depth (m)", color=INK2)
    ax.set_ylabel("signed z error (mm)", color=INK2)
    ax.set_title("Depth error vs. GT depth (marker: median, bar: 25–75th percentile)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, facecolor=SURF)
    print("saved", out)


if __name__ == "__main__":
    main()
