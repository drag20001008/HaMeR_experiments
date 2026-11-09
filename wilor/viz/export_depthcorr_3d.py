"""
depth_correction 결과를 브라우저 3D 뷰어(viewer3d/index.html)용 JSON으로 내보냄.
손마다 viewer3d/data/<seq>/<frame>_h<k>.json:
  vb/va: 보정 전/후 정점 (778*3, mm 정수), jb/ja: 관절 (21*3)
  hp/hc: 손 마스크 안 센서 depth 점 (mm) / 색 (RGB)
  cp/cc: 손 주변(WiLoR 2D bbox 2배 영역, 손 중심 depth +-300mm) 센서 depth 점 / 색
좌표는 color 카메라 OpenCV 좌표 그대로 (뷰어에서 y, z 부호를 뒤집어 표시).
"""
import json
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/nas2/lait/1000_Members/jaehoon/RealSense_D435i_Capture"
DC = os.path.join(ROOT, "depth_correction")
OUT = os.path.join(DC, "viewer3d")
MAX_HAND_PTS, MAX_CTX_PTS = 8000, 8000
rng = np.random.default_rng(0)


def backproject(mask, D, K):
    v, u = np.nonzero(mask)
    z = D[v, u]
    return np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], -1), v, u


def sub(n, k):
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


def mm(a):
    return np.round(np.asarray(a) * 1000).astype(int).reshape(-1).tolist()


os.makedirs(OUT, exist_ok=True)
faces = np.asarray(pickle.load(open("/workspace/drag1008/wilor/mano_data/MANO_RIGHT.pkl", "rb"), encoding="latin1")["f"])
json.dump(faces.astype(int).reshape(-1).tolist(), open(os.path.join(OUT, "faces.json"), "w"))

index = {}
summary = json.load(open(os.path.join(DC, "summary.json")))
for seq in summary:
    K = json.load(open(os.path.join(ROOT, "depth", seq, "intrinsics.json")))["color"]
    R = np.load(os.path.join(DC, seq, "results.npz"))
    recs = [json.loads(r) for r in R["records"]]
    os.makedirs(os.path.join(OUT, "data", seq), exist_ok=True)
    index[seq] = []
    for i, rec in enumerate(recs):
        fr = rec["frame"]
        img = cv2.imread(os.path.join(ROOT, "images", seq, f"{fr}_color.png"))[:, :, ::-1]
        D = cv2.imread(os.path.join(ROOT, "depth", seq, f"{fr}_depth_aligned.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
        H, W = D.shape
        valid = D > 0
        mask = np.zeros((H, W), bool)
        if rec["mask_prompt"] != "none":
            mask = np.load(os.path.join(ROOT, "sam3_seg", seq, f"{fr}.npz"))[f"{rec['mask_prompt']}_masks"][rec["mask_idx"]]

        hp, hv, hu = backproject(mask & valid, D, K)
        k = sub(len(hp), MAX_HAND_PTS)
        hp, hc = hp[k], img[hv[k], hu[k]]

        # 주변 점: WiLoR 2D bbox를 2배로 키운 영역, 손 중심 depth +-300mm
        uv = R["uv_wilor"][i]
        c2, half = (uv.min(0) + uv.max(0)) / 2, (uv.max(0) - uv.min(0)).max()
        x0, y0 = np.clip(np.floor(c2 - half), 0, None).astype(int)
        x1, y1 = np.minimum(np.ceil(c2 + half), [W, H]).astype(int)
        zc = np.median(hp[:, 2]) if len(hp) else R["j_after"][i][:, 2].mean()
        box = np.zeros((H, W), bool)
        box[y0:y1, x0:x1] = True
        ctx = box & valid & ~mask & (np.abs(D - zc) < 0.3)
        cp, cv, cu = backproject(ctx, D, K)
        k = sub(len(cp), MAX_CTX_PTS)
        cp, cc = cp[k], img[cv[k], cu[k]]

        name = f"{fr}_h{rec['hand']}"
        json.dump({"rec": rec,
                   "vb": mm(R["v_before"][i]), "va": mm(R["v_after"][i]),
                   "jb": mm(R["j_before"][i]), "ja": mm(R["j_after"][i]),
                   "hp": mm(hp), "hc": hc.astype(int).reshape(-1).tolist(),
                   "cp": mm(cp), "cc": cc.astype(int).reshape(-1).tolist()},
                  open(os.path.join(OUT, "data", seq, name + ".json"), "w"), separators=(",", ":"))
        index[seq].append({"name": name, "frame": fr, "hand": rec["hand"], "right": rec["right"],
                           "applied": rec["applied"], "shift_mm": round(rec["shift_mm"], 1), "reason": rec["reason"]})
    print(seq, len(index[seq]), flush=True)

json.dump(index, open(os.path.join(OUT, "index.json"), "w"))
print("EXPORT_DONE")
