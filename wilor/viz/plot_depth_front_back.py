"""
손목 depth 오차를 '예측이 GT보다 앞(카메라 쪽, z err < 0)' / '뒤(z err > 0)'로 나눠 본다 (onscreen).
방법별 패널: 앞은 왼쪽, 뒤는 오른쪽으로 |오차| 분포를 그리고, 각 쪽 비율/중앙값/p90을 표시.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [("B_cam_realK", "B: WiLoR cam + real K"),
           ("C_pnp_wbeta", "C: WiLoR β + PnP"),
           ("D_pnp_gtbeta_lfix", "D_lfix: GT β + PnP"),
           ("E_pnp_gtbeta_gt2d_lfix", "E_lfix: GT β + GT 2D (oracle)")]
FRONT, BACK = "#2a78d6", "#eb6834"  # color = side (fixed across panels), reference palette slots 1, 2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
XMAX = 150


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    on = d["onscreen"]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.npz)), "depth_front_back.png")

    print(f"onscreen N={on.sum()}   front: pred이 GT보다 카메라 쪽(z err<0), back: 더 멀리(z err>0)   (mm)")
    print(f"{'method':24s} | {'front %':>7s} {'med':>6s} {'p75':>6s} {'p90':>6s} | {'back %':>7s} {'med':>6s} {'p75':>6s} {'p90':>6s} | "
          f"{'front>20mm':>10s} {'back>20mm':>9s}")
    stats = {}
    for k, lab in METHODS:
        e = d[f"wrist_{k}"][on, 2]
        f, b = -e[e < 0], e[e > 0]
        st = dict(fp=len(f) / len(e) * 100, bp=len(b) / len(e) * 100,
                  fm=np.median(f), f75=np.percentile(f, 75), f90=np.percentile(f, 90),
                  bm=np.median(b), b75=np.percentile(b, 75), b90=np.percentile(b, 90),
                  f20=(f > 20).sum() / len(e) * 100, b20=(b > 20).sum() / len(e) * 100)
        stats[k] = (f, b, st)
        print(f"{k:24s} | {st['fp']:6.1f}% {st['fm']:6.1f} {st['f75']:6.1f} {st['f90']:6.1f} | "
              f"{st['bp']:6.1f}% {st['bm']:6.1f} {st['b75']:6.1f} {st['b90']:6.1f} | {st['f20']:9.1f}% {st['b20']:8.1f}%")

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(10, 11), sharex=True, facecolor=SURF)
    fig.suptitle(f"DexYCB s0 test, onscreen (N={on.sum():,}): is the predicted wrist in front of or behind GT?",
                 color=INK, fontsize=12.5, x=0.02, ha="left")
    bins = np.arange(0, XMAX + 2, 2)
    ctr = (bins[:-1] + bins[1:]) / 2
    ymax = 0
    for k, _ in METHODS:
        f, b, _ = stats[k]
        n = len(f) + len(b)
        ymax = max(ymax, (np.histogram(f, bins)[0] / n * 100).max(), (np.histogram(b, bins)[0] / n * 100).max())
    for ax, (k, lab) in zip(axes, METHODS):
        f, b, st = stats[k]
        n = len(f) + len(b)
        hf = np.histogram(f, bins)[0] / n * 100
        hb = np.histogram(b, bins)[0] / n * 100
        ax.set_facecolor(SURF)
        # 2px surface gap between adjacent bars: width 1.6 on 2mm bins, rounded look via thin edge
        ax.bar(-ctr, hf, width=1.6, color=FRONT, linewidth=0)
        ax.bar(ctr, hb, width=1.6, color=BACK, linewidth=0)
        ax.axvline(0, color=INK, lw=1)
        for med, sgn, c in ((st["fm"], -1, FRONT), (st["bm"], 1, BACK)):
            ax.axvline(sgn * med, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.text(-XMAX + 3, ymax * 0.93, f"in FRONT of GT (nearer camera): {st['fp']:.1f}%\n"
                                        f"median {st['fm']:.1f}  ·  p90 {st['f90']:.1f} mm",
                color=INK, fontsize=9, va="top")
        ax.text(XMAX - 3, ymax * 0.93, f"BEHIND GT (farther): {st['bp']:.1f}%\n"
                                       f"median {st['bm']:.1f}  ·  p90 {st['b90']:.1f} mm",
                color=INK, fontsize=9, va="top", ha="right")
        ax.set_title(lab, color=INK, loc="left", fontsize=10.5)
        ax.set_ylim(0, ymax * 1.05)
        ax.set_xlim(-XMAX, XMAX)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel("% per 2 mm", color=INK2, fontsize=9)
    ticks = np.arange(-150, 151, 25)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{abs(t)}" for t in ticks])
    axes[-1].set_xlabel("depth error magnitude (mm)      ← in front of GT  |  behind GT →      "
                        "(dashed: median of each side; beyond 150 mm not drawn)", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130, facecolor=SURF)
    print("saved", out)


if __name__ == "__main__":
    main()
