"""GT / D_lfix mesh+관절 / 센서 depth point(SAM3 손 마스크)를 3D로 보는 인터랙티브 HTML (샘플당 1개, plotly.js CDN).
샘플: viz_depth_samples.py와 같은 30개 (onscreen 무작위, seed 1). hamer env (extract_depth_anchor 임포트에 pytorch3d 필요).
좌표: 카메라 좌표계 mm. 화면 표시는 (X, Z, -Y) = (오른쪽, depth 안쪽, 위쪽).
"""
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import copy
from torch.utils.data import DataLoader, Subset

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset
from eval_dexycb_pnp_gtbeta import solve_T
from extract_depth_anchor import ROOT, TIPS, REORDER, WithDepth

OUT = "depth_anchor_extract_gtdepth/depth_3d"
OUT_PNG = "depth_anchor_extract_gtdepth/depth_points_topview"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
C_D, C_PT, C_MESH = "#2a78d6", "#1baf7a", "#8f8e87"
SAM3_DIR = f"{ROOT}/sam3_hand_masks"
H, W = 480, 640
MAX_PTS = 3000
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>body{{margin:0;background:#fcfcfb;color:#0b0b0b;font-family:system-ui,sans-serif}}
#h{{padding:10px 14px;font-size:14px}} #h span{{color:#52514e}} #p{{width:100vw;height:calc(100vh - 44px)}}</style></head>
<body><div id="h">{header}</div><div id="p"></div><script>
const D={data};
const faces=D.faces, tr=a=>({{x:a.map(p=>p[0]),y:a.map(p=>p[2]),z:a.map(p=>-p[1])}});
function mesh(v,color,name){{const t=tr(v);return {{type:'mesh3d',...t,i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),k:faces.map(f=>f[2]),
  color:color,opacity:0.35,name:name,showlegend:true,hoverinfo:'skip',flatshading:true}};}}
function skel(j,color,name){{const x=[],y=[],z=[];for(const [a,b] of D.bones){{x.push(j[a][0],j[b][0],null);y.push(j[a][2],j[b][2],null);z.push(-j[a][1],-j[b][1],null);}}
  return {{type:'scatter3d',mode:'lines',x,y,z,line:{{color:color,width:6}},name:name}};}}
function wrist(j,color,name){{return {{type:'scatter3d',mode:'markers',x:[j[0][0]],y:[j[0][2]],z:[-j[0][1]],marker:{{size:5,color:color}},name:name,showlegend:false,
  hovertemplate:name+' wrist<br>X %{{x:.0f}} mm, depth %{{y:.0f}} mm<extra></extra>'}};}}
const c=tr(D.cloud);
const traces=[mesh(D.gt_v,'#9a9a94','GT mesh'),skel(D.gt_j,'#0b0b0b','GT joints'),wrist(D.gt_j,'#0b0b0b','GT'),
  mesh(D.pred_v,'#2a78d6','D_lfix mesh'),skel(D.pred_j,'#2a78d6','D_lfix joints'),wrist(D.pred_j,'#2a78d6','D_lfix'),
  {{type:'scatter3d',mode:'markers',...c,marker:{{size:1.8,color:'#1baf7a',opacity:0.8}},name:'sensor depth points (SAM3 mask)',hoverinfo:'skip'}}];
Plotly.newPlot('p',traces,{{paper_bgcolor:'#fcfcfb',margin:{{l:0,r:0,t:0,b:0}},legend:{{x:0.01,y:0.99}},
  scene:{{aspectmode:'data',xaxis:{{title:'X right (mm)'}},yaxis:{{title:'depth Z (mm)'}},zaxis:{{title:'up (-Y, mm)'}},
  camera:{{eye:{{x:0.0,y:-1.6,z:0.35}},up:{{x:0,y:0,z:1}}}}}}}},{{responsive:true}});
</script></body></html>"""


def skel(ax, pts, color, lw):
    for u, v in BONES:
        ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.plot(pts[0, 0], pts[0, 1], "o", color=color, ms=4.5, zorder=4)


def draw_png(n, row, b, ann, is_right, gj, pj, gv, cloud, ok, K, bsz):
    fx, fy, px, py = K
    fig, (ax_img, ax_top) = plt.subplots(1, 2, figsize=(10, 4.9), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.15]))
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, str(ann["imgname"][row]))), cv2.COLOR_BGR2RGB)
    cx, cy = ann["center"][row]
    half = 0.65 * bsz
    proj = lambda X: np.stack([fx * X[:, 0] / X[:, 2] + px, fy * X[:, 1] / X[:, 2] + py], -1)
    ax_img.imshow(img)
    ax_img.set_xlim(cx - half, cx + half)
    ax_img.set_ylim(cy + half, cy - half)
    ax_img.axis("off")
    skel(ax_img, proj(gj), INK, 2.0)
    skel(ax_img, proj(pj), C_D, 1.2)
    ax_img.set_title("image + 2D reprojection", color=INK2, fontsize=9, loc="left")
    dz = (pj[0, 2] - gj[0, 2]) * 1000
    fig.suptitle(f"#{row} ({'right' if is_right else 'left'} hand)   D_lfix wrist Δz vs GT {dz:+.0f} mm   |   "
                 f"sensor depth points {len(cloud)} (SAM3 mask{'' if ok else ' missing'})",
                 color=INK, fontsize=10.5, x=0.01, ha="left")

    o = gj[0]
    rel = lambda X: ((X - o) * 1000)[:, [0, 2]]
    ax_top.set_facecolor(SURF)
    for s_ in ("top", "right"):
        ax_top.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax_top.spines[s_].set_color(INK2)
    ax_top.tick_params(colors=INK2, labelsize=8)
    ax_top.grid(True, color=GRID, lw=0.8)
    ax_top.set_axisbelow(True)
    g2 = rel(gv)
    if len(cloud):
        c2 = rel(cloud)
        ax_top.scatter(c2[:, 0], c2[:, 1], s=3, c=C_PT, alpha=0.6, lw=0, zorder=2)
    skel(ax_top, rel(gj), INK, 2.0)
    skel(ax_top, rel(pj), C_D, 1.4)
    allp = np.concatenate([rel(gj), rel(pj), g2])
    lo, hi = allp.min(0), allp.max(0)
    c, s = (lo + hi) / 2, max((hi - lo).max() / 2 + 15, 60)
    ax_top.set_xlim(c[0] - s, c[0] + s)
    ax_top.set_ylim(c[1] - s, c[1] + s)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (mm, right →)", color=INK2, fontsize=8.5)
    ax_top.set_ylabel("Z depth (mm, farther ↑)", color=INK2, fontsize=8.5)
    ax_top.set_title("top view (X–Z), origin = GT wrist", color=INK2, fontsize=9, loc="left")
    handles = [plt.Line2D([], [], color=INK, lw=2, label="GT joints"),
               plt.Line2D([], [], color=C_D, lw=1.6, label="D_lfix joints (no depth)"),
               plt.Line2D([], [], ls="", marker="o", ms=4, color=C_PT, label="sensor depth points (SAM3 mask)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(OUT_PNG, f"sample{n:02d}_idx{row}.png"), dpi=110, facecolor=SURF)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_PNG, exist_ok=True)
    ann = np.load(f"{ROOT}/wilor_annotations_v3.npz")
    dev = torch.device("cuda")
    # 같은 30개 샘플
    zs = [np.load(f) for f in sorted(glob.glob("depth_anchor_extract_gtdepth/shard*.npz"))]
    idx_e = np.concatenate([z["idx"] for z in zs])
    on = np.concatenate([z["onscreen"] for z in zs]).astype(bool)
    pick = np.random.default_rng(1).choice(np.where(on)[0], 30, replace=False)
    rows = [int(idx_e[p]) for p in pick]

    # SAM3 마스크 위치
    sam = {}
    for f in sorted(glob.glob(f"{SAM3_DIR}/shard*.npz")):
        z = np.load(f)
        for k, i in enumerate(z["idx"]):
            if int(i) in rows:
                sam[int(i)] = (np.unpackbits(z["mask"][k])[:H * W].reshape(H, W).astype(bool), bool(z["ok"][k]))

    model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt", cfg_path="./pretrained_models/model_config.yaml")
    model = model.to(dev).eval()
    mano_x = copy.deepcopy(model.mano)
    mano_x.shapedirs[:, 0, :] *= -1
    gtL = {s: smplx.MANO(model_path="mano_data_dexycb", is_rhand=(s == "right"), use_pca=False, num_pca_comps=45,
                         flat_hand_mean=False).to(dev) for s in ("right", "left")}
    comps = {s: torch.tensor(np.asarray(L.np_hand_components), dtype=torch.float32, device=dev) for s, L in gtL.items()}
    faces_pred = np.asarray(model.mano.faces, dtype=np.int64)

    ds = ImageDataset(cfg, f"{ROOT}/wilor_annotations_v3.npz", ROOT, train=False, rescale_factor=-1)
    loader = DataLoader(WithDepth(ds, np.array(rows)), batch_size=len(rows), shuffle=False, num_workers=2)
    batch = next(iter(loader))
    to = lambda x: x.to(dev) if isinstance(x, torch.Tensor) else ({k: to(v) for k, v in x.items()} if isinstance(x, dict) else x)
    batch = to(batch)
    links = []
    with torch.no_grad():
        out = model.forward_step(batch, train=False)
        B = len(rows)
        right = batch["right"].float()
        L3 = (right < 0.5).view(B, 1, 1)
        sign = (2 * right - 1).view(B, 1)

        def unm(x):
            x = x.clone()
            x[..., 0] = x[..., 0] * sign
            return x

        fx, fy, px, py = [batch[k].float() for k in ("fx", "fy", "px", "py")]
        betas = batch["mano_params"]["betas"].float()
        pm = out["pred_mano_params"]
        go, hp = pm["global_orient"].reshape(B, -1, 3, 3), pm["hand_pose"].reshape(B, -1, 3, 3)
        o_r = model.mano(global_orient=go, hand_pose=hp, betas=betas, pose2rot=False)
        o_x = mano_x(global_orient=go, hand_pose=hp, betas=betas, pose2rot=False)
        rel_j = torch.where(L3, unm(o_x.joints - o_x.joints[:, :1]), unm(o_r.joints - o_r.joints[:, :1])).float()
        rel_v = torch.where(L3, unm(o_x.vertices - o_x.joints[:, :1]), unm(o_r.vertices - o_r.joints[:, :1])).float()
        uv = batch["box_center"].float()[:, None] + unm(out["pred_keypoints_2d"].float()) * batch["box_size"].float().view(B, 1, 1)
        T = solve_T(rel_j, uv, fx, fy, px, py)
        pred_j, pred_v = (rel_j + T[:, None]).cpu().numpy(), (rel_v + T[:, None]).cpu().numpy()
        pmg = batch["pose_m"].float()
        gv, gj = {}, {}
        for s, Lr in gtL.items():
            o = Lr(global_orient=pmg[:, :3], hand_pose=pmg[:, 3:48] @ comps[s], betas=betas, transl=pmg[:, 48:51])
            gv[s], gj[s] = o.vertices, torch.cat([o.joints[:, :16], o.vertices[:, TIPS[s]]], 1)[:, REORDER]
        gt_v = torch.where(L3, gv["left"], gv["right"]).cpu().numpy()
        gt_j = torch.where(L3, gj["left"], gj["right"]).cpu().numpy()
        gt_faces = {s: np.asarray(gtL[s].faces, dtype=np.int64) for s in gtL}
        D = batch["depth"].cpu().numpy()

    rng = np.random.default_rng(0)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    for n, (b, row) in enumerate(zip(range(B), rows), 1):
        mask, ok = sam.get(row, (np.zeros((H, W), bool), False))
        m = mask & (D[b] > 0)
        Z = D[b][m]
        cloud = np.stack([(uu[m] - float(px[b])) * Z / float(fx[b]), (vv[m] - float(py[b])) * Z / float(fy[b]), Z], -1)
        if len(cloud) > MAX_PTS:
            cloud = cloud[rng.choice(len(cloud), MAX_PTS, replace=False)]
        draw_png(n, row, b, ann, float(right[b]) > .5, gt_j[b], pred_j[b], gt_v[b], cloud, ok,
                 [float(t[b]) for t in (fx, fy, px, py)], batch["box_size"][b].item())
        # GT 좌/우 mesh의 faces 방향이 다르지만 표시엔 무관: 공통 topology라 pred faces 사용
        dz = (pred_j[b][0, 2] - gt_j[b][0, 2]) * 1000
        side = "right" if float(right[b]) > .5 else "left"
        data = dict(faces=faces_pred.tolist(), bones=BONES,
                    gt_v=np.round(gt_v[b] * 1000, 1).tolist(), gt_j=np.round(gt_j[b] * 1000, 1).tolist(),
                    pred_v=np.round(pred_v[b] * 1000, 1).tolist(), pred_j=np.round(pred_j[b] * 1000, 1).tolist(),
                    cloud=np.round(cloud * 1000, 1).tolist())
        header = (f"#{row} ({side} hand) &nbsp; D_lfix wrist Δz vs GT: <b>{dz:+.0f} mm</b> &nbsp; "
                  f"<span>depth points: {len(cloud)} (SAM3 mask {'ok' if ok else 'missing'}) · drag to rotate, scroll to zoom</span>")
        name = f"sample{n:02d}_idx{row}.html"
        with open(os.path.join(OUT, name), "w") as f:
            f.write(HTML.format(title=f"Hand 3D #{row}", header=header, data=json.dumps(data, separators=(",", ":"))))
        links.append((name, row, side, dz, len(cloud)))

    items = "".join(f'<li><a href="{nm}" target="_blank">sample {i:02d} · #{r} ({s}) · D_lfix Δz {dz:+.0f} mm · {c} pts</a></li>'
                    for i, (nm, r, s, dz, c) in enumerate(links, 1))
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Hand 3D samples</title>"
                f"<style>body{{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;padding:16px}}"
                f"li{{margin:4px 0}}a{{color:#2a78d6}}</style></head><body><h3>GT (gray) / D_lfix (blue) / sensor depth points (green)</h3>"
                f"<ol>{items}</ol></body></html>")
    print(f"saved {len(links)} html files + index.html to {OUT}")


if __name__ == "__main__":
    main()
