"""AGILE(논문 그대로 ICP scale + PnP [+ L_joint]) 결과를 WiLoR 기준선 / (b) depth 보정 / oracle과 비교.
AGILE 쪽 손 마스크는 SAM3, (b) 방식은 DexYCB GT seg 마스크(추출 당시 설정)라는 점에 유의."""
import glob

import numpy as np


def load(d):
    zs = [np.load(f) for f in sorted(glob.glob(f"{d}/shard*.npz"))]
    return lambda k: np.concatenate([z[k] for z in zs])


A, S, Gd = load("agile_extract"), load("depth_anchor_extract"), load("depth_anchor_extract_gtdepth")
assert (A("idx") == S("idx")).all() and (S("idx") == Gd("idx")).all()
on = A("onscreen").astype(bool)
G = A("gt_j")
P_c, P_d = A("pnp_j"), S("pred_j")


def b_corr(get, kind, filt):
    med, cnt, cen = get(f"{kind}_med")[:, 0, 0], get(f"{kind}_cnt")[:, 0, 0], get(f"{kind}_cen")[:, 0, 0]
    r = cen / np.linalg.norm(cen, axis=-1, keepdims=True).clip(1e-9)
    use = (cnt >= 1000) & np.isfinite(med)
    if filt:
        use &= (get(f"{kind}_q90")[:, 0, 0] - get(f"{kind}_q10")[:, 0, 0]) <= 100
    return P_d + (np.where(use, med / 1000 / r[:, 2].clip(1e-6), 0)[:, None] * r)[:, None, :]


cj = P_d.mean(1)
rj = cj / np.linalg.norm(cj, axis=-1, keepdims=True)
P_or = P_d + (((G[..., 2].mean(1) - P_d[..., 2].mean(1)) / rj[:, 2])[:, None] * rj)[:, None, :]

rows = [
    ("C: WiLoR β + PnP (= AGILE, s=1)", P_c),
    ("AGILE, sensor depth", A("sensor_agile_j")),
    ("AGILE, sensor depth + L_joint", A("sensor_agile_gn_j")),
    ("AGILE, GT depth", A("gt_agile_j")),
    ("AGILE, GT depth + L_joint", A("gt_agile_gn_j")),
    ("D_lfix: GT β + PnP", P_d),
    ("(b) sensor depth + 필터", b_corr(S, "pred", True)),
    ("(b) GT depth, 실제 손 마스크", b_corr(Gd, "pred", False)),
    ("oracle δ*", P_or),
]


def ev(Pk, m):
    cs = np.linalg.norm(Pk[m] - G[m], axis=-1).mean(-1) * 1000
    e = (Pk[m][:, 0] - G[m][:, 0]) * 1000
    a = np.abs(e)
    return (f"{cs.mean():8.2f} {np.median(cs):7.2f} | {e[:, 0].mean():+5.1f} {e[:, 1].mean():+5.1f} {e[:, 2].mean():+7.1f} | "
            f"{a[:, 0].mean():6.1f} {a[:, 1].mean():6.1f} {a[:, 2].mean():7.1f} | {np.linalg.norm(e, axis=1).mean():7.1f}")


pr, gr = P_c - P_c[:, :1], G - G[:, :1]
s_ref = (pr * gr).sum((1, 2)) / (pr * pr).sum((1, 2))
print("ICP scale (WiLoR 손 크기 배율)   기준값 = GT 관절로 구한 scale")
print(f"  기준값          median {np.median(s_ref):.3f}  p10 {np.percentile(s_ref, 10):.3f}  p90 {np.percentile(s_ref, 90):.3f}")
for src in ("sensor", "gt"):
    s, ok, ni = A(f"{src}_s"), A(f"{src}_icp_ok").astype(bool), A(f"{src}_n_iter")
    print(f"  {src:6s} ICP    median {np.median(s[ok]):.3f}  p10 {np.percentile(s[ok], 10):.3f}  p90 {np.percentile(s[ok], 90):.3f}"
          f" | ICP 수행 {ok.mean() * 100:.1f}% | 30회 미수렴 {(ni[ok] >= 30).mean() * 100:.1f}% | corr w/ 기준 {np.corrcoef(s[ok], s_ref[ok])[0, 1]:.2f}")
print(f"  SAM3 마스크 있음: {A('sam3_ok').mean() * 100:.1f}%")

for name, msk in (("onscreen", on), ("all", np.ones(len(on), bool))):
    print(f"\n[{name}] N={msk.sum()}   손목 오차 mm")
    print(f"  {'method':34s} {'CS':>8s} {'med':>7s} | {'bias x/y/z':>19s} | {'MAE x/y/z':>21s} | {'|e|':>7s}")
    for label, Pk in rows:
        print(f"  {label:34s} {ev(Pk, msk)}")
