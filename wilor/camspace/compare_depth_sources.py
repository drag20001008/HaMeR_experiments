"""(b) depth 보정을 depth 소스별로 비교 (onscreen): 센서 depth vs GT mesh 렌더링 depth(손 마스크 / GT 윤곽).
설정: erode 0, 전체 영역, 중앙값, 유효 픽셀 >= 1000. 필터(분포 폭 q90-q10 <= 100mm) 있음/없음."""
import glob
import numpy as np


def load(d):
    zs = [np.load(f) for f in sorted(glob.glob(f"{d}/shard*.npz"))]
    return lambda k: np.concatenate([z[k] for z in zs])


S, Gd = load("depth_anchor_extract"), load("depth_anchor_extract_gtdepth")
assert (S("idx") == Gd("idx")).all()
on = S("onscreen").astype(bool)
P, G = S("pred_j"), S("gt_j")


def ev(Pk, m):
    cs = np.linalg.norm(Pk[m] - G[m], axis=-1).mean(-1) * 1000
    e = (Pk[m][:, 0] - G[m][:, 0]) * 1000  # 손목 오차 (N,3) mm
    ab = np.abs(e)
    return (f"{cs.mean():6.2f} | bias x/y/z {e[:, 0].mean():+5.1f} {e[:, 1].mean():+5.1f} {e[:, 2].mean():+6.1f} "
            f"| MAE x/y/z {ab[:, 0].mean():5.1f} {ab[:, 1].mean():5.1f} {ab[:, 2].mean():5.1f} "
            f"| |e| {np.linalg.norm(e, axis=1).mean():5.1f}")


def correct(get, kind, spread_max=None, e=0):
    med, cnt, cen = get(f"{kind}_med")[:, e, 0], get(f"{kind}_cnt")[:, e, 0], get(f"{kind}_cen")[:, e, 0]
    spread = get(f"{kind}_q90")[:, e, 0] - get(f"{kind}_q10")[:, e, 0]
    r = cen / np.linalg.norm(cen, axis=-1, keepdims=True).clip(1e-9)
    use = (cnt >= 1000) & np.isfinite(med)
    if spread_max:
        use &= spread <= spread_max
    dm = np.where(use, med / 1000 / r[:, 2].clip(1e-6), 0)
    return P + (dm[:, None] * r)[:, None, :], use


cj = P.mean(1)
rj = cj / np.linalg.norm(cj, axis=-1, keepdims=True)
P_or = P + (((G[..., 2].mean(1) - P[..., 2].mean(1)) / rj[:, 2])[:, None] * rj)[:, None, :]

print(f"onscreen N={on.sum()}   (손목 오차, mm)                          CS |")
print(f"  {'D_lfix (depth 안 씀)':44s} {ev(P, on)}")
for label, get, kind in (("센서 depth", S, "pred"),
                         ("GT depth, 실제 손 마스크 기준", Gd, "pred"),
                         ("GT depth, GT 손 윤곽 기준 (가림 없음)", Gd, "predsil")):
    for sm in (None, 100):
        Pk, use = correct(get, kind, sm)
        tag = f"{label} {'+ 필터' if sm else ''}"
        print(f"  {tag:44s} {ev(Pk, on)} | 보정 {use[on].mean() * 100:5.1f}%")
print(f"  {'oracle δ* (b 방식 상한)':44s} {ev(P_or, on)}")
