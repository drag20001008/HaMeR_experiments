"""depth 보정 (b)에 GT 없이 쓸 수 있는 필터: |delta| 상한, 픽셀 delta 분포 폭(q90-q10) 상한. 걸리면 보정 안 함(D_lfix 유지)."""
import glob
import numpy as np

zs = [np.load(f) for f in sorted(glob.glob("depth_anchor_extract/shard*.npz"))]
cat = lambda k: np.concatenate([z[k] for z in zs])
on = cat("onscreen").astype(bool)
P, G = cat("pred_j"), cat("gt_j")


def ev(Pk, m):
    cs = np.linalg.norm(Pk[m] - G[m], axis=-1).mean(-1) * 1000
    ez = (Pk[m][:, 0, 2] - G[m][:, 0, 2]) * 1000
    return cs.mean(), np.median(cs), ez.mean(), np.abs(ez).mean(), (ez > 0).mean() * 100


print("onscreen N=%d   D_lfix: CS %.2f med %.2f | z bias %+.1f MAE %.1f | behind %.1f%%" % (on.sum(), *ev(P, on)))
print("erode  bias  |d|max  spread |     CS    med | z bias  z MAE | behind | corrected")
for e in (0, 2):
    med, cnt, cen = cat("pred_med")[:, e, 0], cat("pred_cnt")[:, e, 0], cat("pred_cen")[:, e, 0]
    spread = cat("pred_q90")[:, e, 0] - cat("pred_q10")[:, e, 0]
    r = cen / np.linalg.norm(cen, axis=-1, keepdims=True).clip(1e-9)
    for bias in (0.0, 9.7):
        for dmax in (None, 80, 50, 30):
            for smax in (None, 100, 60):
                d = med - bias
                use = (cnt >= 1000) & np.isfinite(med)
                if dmax:
                    use &= np.abs(d) <= dmax
                if smax:
                    use &= spread <= smax
                dm = np.where(use, d / 1000 / r[:, 2].clip(1e-6), 0)
                cs, cm, zb, zm, bh = ev(P + (dm[:, None] * r)[:, None, :], on)
                print(f"{e:5d} {bias:5.1f} {dmax or '-':>6} {smax or '-':>6} | {cs:6.2f} {cm:6.2f} | {zb:+6.1f} {zm:6.1f} | {bh:5.1f}% | {use[on].mean() * 100:5.1f}%")
