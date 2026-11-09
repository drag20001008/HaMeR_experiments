"""
extract_depth_anchor.py 결과(shard*.npz)로 depth anchor(표면 대 표면) 분석.

 [1] 신뢰도  : GT mesh 표면 vs 센서 depth (픽셀 전역 히스토그램 + 샘플별 중앙값), 가시/가림/depth 결측 비율
 [2][3] 후보 : 영역(all/palm/fingers) x erosion(0/2/4) x 집계(med/mean/tmean) x 최소 픽셀 수 k
               보정 (b): 사용 픽셀(pred 표면)의 3D 중심 c를 지나는 ray r=c/|c| 로 강체 평행이동
               delta = agg(D - z_pred) / r_z,  P' = P + delta * r   (유효 픽셀 < k 면 보정 안 함)
               raw(센서 그대로) / sensor-bias 보정(GT에서 잰 센서 편향 중앙값을 뺌, GT 통계 사용이므로 참고용) 둘 다
 [6] 기준    : lower = D_lfix, oracle = 같은 (b) 방식으로 GT 관절 평균 z에 정확히 맞춘 delta*, floor = GT 손목 위치 사용
               gap closure = (CS_D - CS_cand) / (CS_D - CS_oracle)
 [7] 조건별  : 가림(렌더 픽셀 중 손 마스크 비율), GT depth 구간, 좌/우, 물체 가림 비율
"""
import argparse
import glob
import json
import os

import numpy as np

AGGS = ("med", "mean", "tmean")
KS = (1, 200, 1000)


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "shard*.npz")))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    R = {}
    for k in zs[0].files:
        if k in ("hist", "hist_out"):
            R[k] = sum(z[k] for z in zs)
        elif k in ("hist_edges", "erodes", "regions"):
            R[k] = zs[0][k]
        else:
            R[k] = np.concatenate([z[k] for z in zs])
    print(f"loaded {len(fs)} shards, N={len(R['idx'])}")
    return R


def metrics(P, G, mask):
    P, G = P[mask], G[mask]
    cs = np.linalg.norm(P - G, axis=-1).mean(-1) * 1000
    e = (P[:, 0] - G[:, 0]) * 1000
    ez = e[:, 2]
    return dict(N=int(mask.sum()), CS=float(cs.mean()), CS_med=float(np.median(cs)),
                wz_bias=float(ez.mean()), wz_mae=float(np.abs(ez).mean()), wz_med=float(np.median(ez)),
                w_norm=float(np.linalg.norm(e, axis=1).mean()),
                behind=float((ez > 0).mean() * 100), wx_mae=float(np.abs(e[:, 0]).mean()), wy_mae=float(np.abs(e[:, 1]).mean()))


def shift_along(P, c, delta_m, use):
    r = c / np.linalg.norm(c, axis=-1, keepdims=True).clip(1e-9)
    d = np.where(use, delta_m / r[:, 2].clip(1e-6), 0.0)
    return P + (d[:, None] * r)[:, None, :]


def fmt(m, extra=""):
    return (f"{m['CS']:7.2f} {m['CS_med']:7.2f} | z bias {m['wz_bias']:+6.1f} med {m['wz_med']:+6.1f} MAE {m['wz_mae']:6.1f} | "
            f"x/y MAE {m['wx_mae']:5.1f}/{m['wy_mae']:5.1f} | |e| {m['w_norm']:6.1f} | behind {m['behind']:5.1f}% {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./depth_anchor_extract")
    args = ap.parse_args()
    R = load(args.dir)
    E, RG = list(R["erodes"]), list(R["regions"])
    P, G = R["pred_j"], R["gt_j"]
    N = len(P)
    on = R["onscreen"].astype(bool)
    subsets = {"all": np.ones(N, bool), "onscreen": on}
    out = {}

    # ---------------- [1] reliability ----------------
    print("\n" + "=" * 120 + "\n[1] GT mesh surface vs sensor depth   (delta = D_sensor - z_GTsurface, +면 센서가 더 멀리 잼)")
    print(f"sanity: GT mesh recon err vs label joints max {R['gt_recon_err'].max():.4f} mm")
    edges = R["hist_edges"]
    ctr = (edges[:-1] + edges[1:]) / 2
    rel = {}
    for ei, e in enumerate(E):
        for ri, rg in enumerate(RG):
            h = R["hist"][ei, ri]
            lo, hi = R["hist_out"][ei, ri]
            tot = h.sum() + lo + hi
            cdf = np.cumsum(h) / tot + lo / tot
            med = float(np.interp(0.5, cdf, ctr))
            within = {t: float(h[np.abs(ctr) <= t].sum() / tot * 100) for t in (5, 10, 20)}
            s_med = R["gt_med"][:, ei, ri]
            s_med = s_med[np.isfinite(s_med) & (R["gt_cnt"][:, ei, ri] > 0)]
            rel[f"e{e}_{rg}"] = dict(pixel_median=med, pixel_within5=within[5], pixel_within10=within[10],
                                     pixel_within20=within[20], pixel_out100=float((lo + hi) / tot * 100),
                                     sample_median_of_medians=float(np.median(s_med)),
                                     sample_iqr=[float(np.percentile(s_med, 25)), float(np.percentile(s_med, 75))])
            print(f"  erode {e} {rg:8s}: pixel median {med:+6.1f} mm | |delta|<=5/10/20mm: {within[5]:5.1f}/{within[10]:5.1f}/{within[20]:5.1f}% "
                  f"| >100mm {(lo + hi) / tot * 100:4.1f}% | per-sample median: med {np.median(s_med):+6.1f} IQR [{np.percentile(s_med, 25):+5.1f},{np.percentile(s_med, 75):+5.1f}]")
    nr = np.maximum(R["gt_n_render"], 1)
    vis = {"hand_mask/rendered": R["gt_n_render_hand"] / nr, "object/rendered": R["gt_n_render_obj"] / nr,
           "background/rendered": R["gt_n_render_bg"] / nr,
           "no_depth/(rendered&hand)": R["gt_n_render_hand_nodepth"] / np.maximum(R["gt_n_render_hand"], 1)}
    print("  GT silhouette composition (per-sample median / p10 / p90):")
    for k, v in vis.items():
        print(f"     {k:26s} {np.median(v) * 100:5.1f}% / {np.percentile(v, 10) * 100:5.1f}% / {np.percentile(v, 90) * 100:5.1f}%")
    out["reliability"] = rel
    sensor_bias = rel["e2_all"]["sample_median_of_medians"]
    print(f"  -> sensor bias used for 'biascorr' variants (e2_all per-sample median): {sensor_bias:+.2f} mm")

    # ---------------- [6] baselines ----------------
    print("\n" + "=" * 120 + "\n[6] baselines")
    base = {"D_lfix (no depth)": P}
    cj = P.mean(1)
    rj = cj / np.linalg.norm(cj, axis=-1, keepdims=True)
    dstar = (G[..., 2].mean(1) - P[..., 2].mean(1)) / rj[:, 2]
    base["oracle delta* (b)"] = P + (dstar[:, None] * rj)[:, None, :]
    base["floor: GT wrist pos"] = P - P[:, :1] + G[:, :1]
    bm = {}
    for sname, msk in subsets.items():
        print(f"  [{sname}] N={msk.sum()}   {'CS':>7s} {'CS_med':>7s}")
        for k, Pk in base.items():
            m = metrics(Pk, G, msk)
            bm[(sname, k)] = m
            print(f"   {k:24s} {fmt(m)}")
    out["baselines"] = {f"{s}|{k}": v for (s, k), v in bm.items()}

    # ---------------- [2][3] candidates ----------------
    print("\n" + "=" * 120 + "\n[2][3] candidates: (b) shift along ray through centroid of used pred-surface pixels")
    cands = {}
    for variant, bias in (("raw", 0.0), ("biascorr", sensor_bias)):
        for ei, e in enumerate(E):
            for ri, rg in enumerate(RG):
                for a in AGGS:
                    for k in KS:
                        cnt = R["pred_cnt"][:, ei, ri]
                        val = R[f"pred_{a}"][:, ei, ri]
                        use = (cnt >= k) & np.isfinite(val)
                        dm = np.where(use, (val - bias) / 1000.0, 0.0)
                        cands[(variant, e, rg, a, k)] = (shift_along(P, R["pred_cen"][:, ei, ri], dm, use), use)
    table = []
    for key, (Pk, use) in cands.items():
        row = {"key": key, "fallback_all": float((~use).mean() * 100)}
        for sname, msk in subsets.items():
            m = metrics(Pk, G, msk)
            lo_cs, or_cs = bm[(sname, "D_lfix (no depth)")]["CS"], bm[(sname, "oracle delta* (b)")]["CS"]
            m["gap_closure"] = (lo_cs - m["CS"]) / max(lo_cs - or_cs, 1e-9) * 100
            row[sname] = m
        table.append(row)
    table.sort(key=lambda r: r["onscreen"]["CS"])
    print(f"  top 15 by onscreen CS-MPJPE   (variant, erode, region, agg, min_pixels)")
    for r in table[:15]:
        m = r["onscreen"]
        extra = f"| gap {m['gap_closure']:5.1f}% | fallback {r['fallback_all']:4.1f}%"
        print(f"   {str(r['key']):42s} {fmt(m, extra)}")
    # 요인별 평균 효과 (onscreen CS)
    print("  factor marginals (median onscreen CS over other factors; 일부 후보가 크게 튀어서 평균 대신 중앙값):")
    for fi, fname in enumerate(("variant", "erode", "region", "agg", "min_pixels")):
        vals = sorted({r["key"][fi] for r in table}, key=str)
        s = "  ".join(f"{v}: {np.median([r['onscreen']['CS'] for r in table if r['key'][fi] == v]):.2f}" for v in vals)
        print(f"     {fname:10s} {s}")
    out["candidates"] = [{"key": list(map(str, r["key"])), "fallback_all": r["fallback_all"],
                          **{s: r[s] for s in subsets}} for r in table]

    # ---------------- [7] stratified ----------------
    print("\n" + "=" * 120 + "\n[7] stratified (onscreen): D_lfix vs oracle vs best raw / best biascorr")
    best = {v: next(r for r in table if r["key"][0] == v)["key"] for v in ("raw", "biascorr")}
    methods = {"D_lfix": P, "oracle": base["oracle delta* (b)"]}
    for v, k in best.items():
        methods[f"best_{v} {k[1:]}"] = cands[k][0]
    hand_frac = R["pred_n_render_hand"] / np.maximum(R["pred_n_render"], 1)
    obj_frac = R["gt_n_render_obj"] / np.maximum(R["gt_n_render"], 1)
    gz = G[:, 0, 2]
    strata = {
        "pred silhouette ∩ hand mask": [(f"{a:.1f}-{b:.1f}", (hand_frac >= a) & (hand_frac < b)) for a, b in
                                        ((0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01))],
        "GT silhouette occluded by object": [(f"{a:.2f}-{b:.2f}", (obj_frac >= a) & (obj_frac < b)) for a, b in
                                             ((0, .01), (.01, .05), (.05, .15), (.15, .3), (.3, 1.01))],
        "GT wrist depth (m)": [(f"{a:.2f}-{b:.2f}", (gz >= a) & (gz < b)) for a, b in
                               ((0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 9))],
        "hand side": [("right", R["right"] > .5), ("left", R["right"] < .5)],
    }
    out["stratified"] = {}
    for sname, bins in strata.items():
        print(f"  -- {sname}")
        print(f"     {'bin':12s} {'N':>6s} " + " ".join(f"{m[:24]:>24s}" for m in methods))
        for bname, bm_ in bins:
            msk = bm_ & on
            if msk.sum() < 20:
                continue
            ms = {m: metrics(Pk, G, msk) for m, Pk in methods.items()}
            out["stratified"][f"{sname}|{bname}"] = ms
            print(f"     {bname:12s} {msk.sum():6d} " + " ".join(
                f"{ms[m]['CS']:9.1f} (z{ms[m]['wz_bias']:+6.1f}){'':6s}" for m in methods))

    with open(os.path.join(args.dir, "analysis_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nsaved {os.path.join(args.dir, 'analysis_summary.json')}")


if __name__ == "__main__":
    main()
