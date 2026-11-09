"""최종 설정(erode 0, all, median, >=1000px, spread<=100mm) 오차 요인 분해 (onscreen)."""
import glob
import numpy as np

zs = [np.load(f) for f in sorted(glob.glob("depth_anchor_extract/shard*.npz"))]
cat = lambda k: np.concatenate([z[k] for z in zs])
on = cat("onscreen").astype(bool)
P, G = cat("pred_j"), cat("gt_j")
e = 0
med, cnt, cen = cat("pred_med")[:, e, 0], cat("pred_cnt")[:, e, 0], cat("pred_cen")[:, e, 0]
spread = cat("pred_q90")[:, e, 0] - cat("pred_q10")[:, e, 0]
g = cat("gt_med")[:, e, 0]  # 샘플별 센서 depth - GT 표면 (센서/라벨 불일치)
r = cen / np.linalg.norm(cen, axis=-1, keepdims=True).clip(1e-9)
valid = (cnt >= 1000) & np.isfinite(med) & np.isfinite(g)
gate = valid & (spread <= 100)

cj = P.mean(1)
rj = cj / np.linalg.norm(cj, axis=-1, keepdims=True)
dstar = (G[..., 2].mean(1) - P[..., 2].mean(1)) / rj[:, 2]
P_or = P + (dstar[:, None] * rj)[:, None, :]


def shift(d_mm, use):
    dm = np.where(use, d_mm / 1000 / r[:, 2].clip(1e-6), 0)
    return P + (dm[:, None] * r)[:, None, :]


def ev(Pk, m):
    cs = np.linalg.norm(Pk[m] - G[m], axis=-1).mean(-1) * 1000
    ez = (Pk[m][:, 0, 2] - G[m][:, 0, 2]) * 1000
    ra = np.linalg.norm((Pk[m] - Pk[m][:, :1]) - (G[m] - G[m][:, :1]), axis=-1).mean(-1) * 1000
    return f"CS {cs.mean():6.2f} (med {np.median(cs):6.2f}) | z bias {ez.mean():+6.1f} MAE {np.abs(ez).mean():6.1f} | RA {ra.mean():5.2f}"


rows = [
    ("D_lfix (depth 안 씀)", P, on),
    ("최종: 센서 depth + 필터", shift(med, gate), on),
    ("  └ 보정된 샘플만", shift(med, gate), on & gate),
    ("  └ 필터로 보정 안 된 샘플만", P, on & ~gate),
    ("최종 + 센서 오차만 GT로 제거 (필터 유지)", shift(med - g, gate), on),
    ("센서 오차 GT로 제거 + 필터 없이 전부 보정", shift(med - g, valid), on),
    ("oracle δ* (b 방식 상한)", P_or, on),
]
print(f"onscreen N={on.sum()}  corrected by final gate: {gate[on].mean() * 100:.1f}%")
for name, Pk, m in rows:
    print(f"  {name:40s} N={m.sum():6d}  {ev(Pk, m)}")
d_err = (med - dstar * rj[:, 2] * 1000)[on & gate]   # z 기준 보정량 오차 (mm)
print("\n보정된 샘플의 보정량 오차 분해 (mm, z 기준): 추정 median(Δ) - oracle")
print(f"  전체 오차          : bias {d_err.mean():+6.2f}  std {d_err.std():6.2f}  p5/p50/p95 {np.percentile(d_err, [5, 50, 95]).round(1)}")
gs = g[on & gate]
print(f"  ├ 센서/라벨 불일치 : bias {gs.mean():+6.2f}  std {gs.std():6.2f}  p5/p50/p95 {np.percentile(gs, [5, 50, 95]).round(1)}")
res = d_err - gs
print(f"  └ 방법 자체 잔차   : bias {res.mean():+6.2f}  std {res.std():6.2f}  p5/p50/p95 {np.percentile(res, [5, 50, 95]).round(1)}")
print(f"  분산 비율: 센서 {gs.var() / d_err.var() * 100:.0f}%  방법 {res.var() / d_err.var() * 100:.0f}%  공분산항 {2 * np.cov(gs, res)[0, 1] / d_err.var() * 100:+.0f}%")
