"""DexYCB 라벨 pose_m(PCA45 + trans)으로 GT MANO mesh를 재현하고, 라벨 joint_3d와 비교.
smplx는 num_pca_comps==45면 use_pca를 몰래 끄므로(axis-angle로 해석), PCA 디코딩을 직접 한다:
  hand_aa = coeffs @ hands_components  (+ hands_mean은 flat_hand_mean=False로 smplx가 더함) -> manopth와 동일."""
import os
import numpy as np
import torch
import smplx

ROOT = "/nas2/lait/1000_Members/jaehoon/DexYCB"
# manopth fingertip vertex ids (DexYCB 라벨과 대조해 확인: 왼손 중지 끝만 444가 아니라 445)
TIPS = {"right": [745, 317, 444, 556, 673], "left": [745, 317, 445, 556, 673]}
REORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

d = np.load(f"{ROOT}/wilor_annotations_v3.npz")
test = np.where(d["official_split"] == "test")[0]
rng = np.random.default_rng(0)
ids = np.concatenate([rng.choice(test[d["right"][test] > .5], 100, replace=False),
                      rng.choice(test[d["right"][test] < .5], 100, replace=False)])
layers = {s: smplx.MANO(model_path="mano_data_dexycb", is_rhand=(s == "right"), use_pca=False, num_pca_comps=45,
                        flat_hand_mean=False) for s in ("right", "left")}
comps = {s: torch.tensor(np.asarray(L.np_hand_components), dtype=torch.float32) for s, L in layers.items()}
res = {"right": [], "left": []}
for i in ids:
    side = "right" if d["right"][i] > .5 else "left"
    lab = np.load(os.path.join(ROOT, d["imgname"][i].replace("color_", "labels_").replace(".jpg", ".npz")))
    p = torch.from_numpy(lab["pose_m"][:1]).float()
    o = layers[side](global_orient=p[:, :3], hand_pose=p[:, 3:48] @ comps[side],
                     betas=torch.from_numpy(d["betas"][i:i + 1]), transl=p[:, 48:51])
    j = torch.cat([o.joints[:, :16], o.vertices[:, TIPS[side]]], 1)[:, REORDER][0].detach().numpy()
    res[side].append(np.linalg.norm(j - lab["joint_3d"][0], axis=1) * 1000)
for s, r in res.items():
    r = np.array(r)
    print(f"{s:5s} N={len(r)}  vs label joint_3d: mean {r.mean():.4f}  max {r.max():.4f} mm")
