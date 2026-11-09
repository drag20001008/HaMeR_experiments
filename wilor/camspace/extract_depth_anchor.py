"""
DexYCB test: depth anchor(표면 대 표면) 분석용 데이터 추출. shard 단위로 병렬 실행 (hamer env: pytorch3d 필요).

샘플마다
  - pred mesh: D_lfix (GT beta + WiLoR theta/R, 왼손은 shapedirs x 보정, WiLoR 2D + 실제 K로 closed-form PnP)
  - GT mesh  : 라벨 pose_m (PCA45 직접 디코딩, check_gt_mesh.py와 동일 재현, 라벨 joint_3d와 0.0002mm 일치)
를 실제 K로 래스터화해 픽셀별 표면 z(zbuf)를 얻고, 센서 depth D와의 차이 delta = D - zbuf (mm)를
  렌더된 픽셀 & 손 마스크(seg==255, erosion e px) & depth 유효 인 픽셀에서
  영역(all / palm / fingers) x erosion(0/2/4) 별로 집계(개수, 중앙값, 평균, 10% trimmed mean, q10/q90, 사용 픽셀 3D 중심)해 저장.
GT mesh 쪽은 픽셀 delta 전역 히스토그램도 누적 (분석 1: 표면 depth 신뢰도).
"""
import argparse
import copy
import os

import numpy as np
import smplx
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from torch.utils.data import DataLoader, Dataset

from wilor.models import load_wilor
from wilor.datasets.image_dataset import ImageDataset
from eval_metric_scale import split_by_official
from eval_dexycb_pnp_gtbeta import solve_T

ROOT = "/nas2/lait/1000_Members/jaehoon/DexYCB"
TIPS = {"right": [745, 317, 444, 556, 673], "left": [745, 317, 445, 556, 673]}
REORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
REGIONS = ("all", "palm", "fingers")
ERODES = (0, 2, 4)
HIST_EDGES = np.arange(-100, 101, 1.0)  # mm


class WithDepth(Dataset):
    def __init__(self, base, idxs):
        self.base, self.idxs = base, idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        item = self.base[self.idxs[k]]
        folder, fname = os.path.split(item["imgname_rel"])
        num = fname.replace("color_", "").replace(".jpg", "")
        dp = os.path.join(ROOT, folder, f"aligned_depth_to_color_{num}.png")
        depth = np.array(Image.open(dp)).astype(np.float32) * 0.001 if os.path.exists(dp) else np.zeros((480, 640), np.float32)
        lab = np.load(os.path.join(ROOT, folder, f"labels_{num}.npz"))
        item["depth"] = depth
        item["seg"] = lab["seg"].astype(np.uint8)
        item["pose_m"] = lab["pose_m"][0].astype(np.float32)
        return item


def erode(mask, r):
    if r == 0:
        return mask
    return ~(F.max_pool2d((~mask).float().unsqueeze(1), 2 * r + 1, 1, r).squeeze(1) > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_npz", default=f"{ROOT}/wilor_annotations_v3.npz")
    p.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    p.add_argument("--checkpoint", default="./pretrained_models/wilor_final.ckpt")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--out_dir", default="./depth_anchor_extract")
    p.add_argument("--depth_source", choices=["sensor", "gt"], default="sensor",
                   help="gt: 센서 depth 대신 GT mesh 렌더링 depth를 depth map으로 사용 (완벽한 depth 가정). "
                        "이때 kind 'pred'=실제 손 마스크 기준, 'predsil'=GT 손 윤곽 기준")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")

    model, cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.model_config)
    model = model.to(dev).eval()
    mano_xflip = copy.deepcopy(model.mano)  # 왼손용: shapedirs x 부호 반전 (lfix)
    mano_xflip.shapedirs[:, 0, :] *= -1
    gt_layers = {s: smplx.MANO(model_path="mano_data_dexycb", is_rhand=(s == "right"), use_pca=False,
                               num_pca_comps=45, flat_hand_mean=False).to(dev) for s in ("right", "left")}
    gt_comps = {s: torch.tensor(np.asarray(L.np_hand_components), dtype=torch.float32, device=dev)
                for s, L in gt_layers.items()}

    # faces + 면별 영역 라벨 (skinning weight argmax == 0(root) -> palm, 나머지 -> fingers)
    faces = {"pred": torch.as_tensor(np.asarray(model.mano.faces, dtype=np.int64), device=dev),
             "right": torch.as_tensor(np.asarray(gt_layers["right"].faces, dtype=np.int64), device=dev),
             "left": torch.as_tensor(np.asarray(gt_layers["left"].faces, dtype=np.int64), device=dev)}
    vert_part = model.mano.lbs_weights.argmax(1)  # (778,)
    face_is_palm = (vert_part[faces["pred"]] == 0).sum(1) >= 2  # (F,) 같은 topology라 좌우 공통
    NF = faces["pred"].shape[0]

    ds_full = ImageDataset(cfg, args.dataset_npz, ROOT, train=False, rescale_factor=-1)
    idx_all = split_by_official(ds_full, "test")[args.shard::args.n_shards]
    print(f"shard {args.shard}/{args.n_shards}: {len(idx_all)} samples", flush=True)
    loader = DataLoader(WithDepth(ds_full, idx_all), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    H, W = 480, 640
    raster = MeshRasterizer(raster_settings=RasterizationSettings(image_size=(H, W), blur_radius=0.0,
                                                                  faces_per_pixel=1, bin_size=0))
    uu, vv = torch.meshgrid(torch.arange(W, device=dev, dtype=torch.float32),
                            torch.arange(H, device=dev, dtype=torch.float32), indexing="xy")  # (H,W)

    keys_ps = ["idx", "right", "onscreen", "box_size", "pred_j", "gt_j", "T", "gt_recon_err",
               "n_hand_mask", "n_depth_valid_hand"]
    acc = {k: [] for k in keys_ps}
    KINDS = ("pred", "gt") if args.depth_source == "sensor" else ("pred", "predsil")
    for kind in KINDS:
        for k in ("n_render", "n_render_hand", "n_render_obj", "n_render_bg", "n_render_hand_nodepth"):
            acc[f"{kind}_{k}"] = []
        for k in ("cnt", "med", "mean", "tmean", "q10", "q90", "cen"):
            acc[f"{kind}_{k}"] = []
    hist = np.zeros((len(ERODES), len(REGIONS), len(HIST_EDGES) - 1), np.int64)
    hist_out = np.zeros((len(ERODES), len(REGIONS), 2), np.int64)  # (< -100, > 100)

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    for bi, batch in enumerate(loader):
        if 0 < args.max_batches <= bi:
            break
        batch = to_dev(batch)
        with torch.no_grad():
            out = model.forward_step(batch, train=False)
            B = out["pred_cam"].shape[0]
            right = batch["right"].float()
            is_left = (right < 0.5)
            sign = (2 * right - 1).view(B, 1)

            def unm(x):
                x = x.clone()
                x[..., 0] = x[..., 0] * sign
                return x

            fx, fy, px, py = [batch[k].float() for k in ("fx", "fy", "px", "py")]
            bc, bsz = batch["box_center"].float(), batch["box_size"].float()
            betas = batch["mano_params"]["betas"].float()
            gt3d = unm(batch["keypoints_3d"][..., :3].float())
            gt2d = batch["orig_keypoints_2d"][..., :2].float()
            onscreen = ((gt2d[..., 0] >= 0) & (gt2d[..., 0] < W) & (gt2d[..., 1] >= 0) & (gt2d[..., 1] < H)).all(1)

            # ---- pred mesh: D_lfix ----
            pm = out["pred_mano_params"]
            go, hp = pm["global_orient"].reshape(B, -1, 3, 3), pm["hand_pose"].reshape(B, -1, 3, 3)
            o_r = model.mano(global_orient=go, hand_pose=hp, betas=betas, pose2rot=False)
            o_x = mano_xflip(global_orient=go, hand_pose=hp, betas=betas, pose2rot=False)
            L3 = is_left.view(B, 1, 1)
            rel_j = torch.where(L3, unm(o_x.joints - o_x.joints[:, :1]), unm(o_r.joints - o_r.joints[:, :1])).float()
            rel_v = torch.where(L3, unm(o_x.vertices - o_x.joints[:, :1]), unm(o_r.vertices - o_r.joints[:, :1])).float()
            uv_w = bc[:, None] + unm(out["pred_keypoints_2d"].float()) * bsz.view(B, 1, 1)
            T = solve_T(rel_j, uv_w, fx, fy, px, py)
            pred_j, pred_v = rel_j + T[:, None], rel_v + T[:, None]

            # ---- GT mesh (camera frame, 좌우 각자 모델) ----
            pmg = batch["pose_m"].float()
            gv, gj = {}, {}
            for s, Lr in gt_layers.items():
                o = Lr(global_orient=pmg[:, :3], hand_pose=pmg[:, 3:48] @ gt_comps[s], betas=betas, transl=pmg[:, 48:51])
                gv[s] = o.vertices
                gj[s] = torch.cat([o.joints[:, :16], o.vertices[:, TIPS[s]]], 1)[:, REORDER]
            gt_v = torch.where(L3, gv["left"], gv["right"]).float()
            gt_j_recon = torch.where(L3, gj["left"], gj["right"]).float()

            # ---- rasterize ----
            Kmat = torch.zeros(B, 3, 3, device=dev)
            Kmat[:, 0, 0], Kmat[:, 1, 1], Kmat[:, 0, 2], Kmat[:, 1, 2], Kmat[:, 2, 2] = fx, fy, px, py, 1
            cams = cameras_from_opencv_projection(torch.eye(3, device=dev).expand(B, 3, 3), torch.zeros(B, 3, device=dev),
                                                  Kmat, torch.tensor([[H, W]], device=dev).expand(B, 2))
            gt_faces = [faces["left"] if l else faces["right"] for l in is_left.tolist()]
            rendered = {}
            for mk, V, Fl in (("pred", pred_v, [faces["pred"]] * B), ("gt", gt_v, gt_faces)):
                frag = raster(Meshes(verts=list(V), faces=Fl), cameras=cams)
                rendered[mk] = (frag.zbuf[..., 0], frag.pix_to_face[..., 0])
            zb_gt = rendered["gt"][0]
            D = batch["depth"].float() if args.depth_source == "sensor" else torch.where(zb_gt > 0, zb_gt, torch.zeros_like(zb_gt))
            seg = batch["seg"]
            hand = seg == 255
            obj = (seg != 255) & (seg != 0)
            dvalid = D > 0
            acc["n_hand_mask"].append(hand.sum((1, 2)).cpu().numpy())
            acc["n_depth_valid_hand"].append((hand & dvalid).sum((1, 2)).cpu().numpy())

            # (kind, 어느 mesh를 D와 비교하나, 손 영역 마스크)
            if args.depth_source == "sensor":
                jobs = (("pred", "pred", hand), ("gt", "gt", hand))
            else:
                jobs = (("pred", "pred", hand), ("predsil", "pred", zb_gt > 0))
            for kind, mk, hmask in jobs:
                zb, p2f = rendered[mk]
                rend = zb > 0
                palm = rend & face_is_palm[(p2f.clamp(min=0) % NF)]
                acc[f"{kind}_n_render"].append(rend.sum((1, 2)).cpu().numpy())
                acc[f"{kind}_n_render_hand"].append((rend & hand).sum((1, 2)).cpu().numpy())
                acc[f"{kind}_n_render_obj"].append((rend & obj).sum((1, 2)).cpu().numpy())
                acc[f"{kind}_n_render_bg"].append((rend & (seg == 0)).sum((1, 2)).cpu().numpy())
                acc[f"{kind}_n_render_hand_nodepth"].append((rend & hand & ~dvalid).sum((1, 2)).cpu().numpy())
                delta = (D - zb) * 1000.0  # mm
                Xs, Ys = (uu - px.view(B, 1, 1)) * zb / fx.view(B, 1, 1), (vv - py.view(B, 1, 1)) * zb / fy.view(B, 1, 1)
                res = {k: np.zeros((B, len(ERODES), len(REGIONS)), np.float32) for k in ("cnt", "med", "mean", "tmean", "q10", "q90")}
                cen = np.zeros((B, len(ERODES), len(REGIONS), 3), np.float32)
                for ei, e in enumerate(ERODES):
                    base = rend & erode(hmask, e) & dvalid
                    for ri, rg in enumerate(REGIONS):
                        m = base if rg == "all" else (base & palm if rg == "palm" else base & ~palm)
                        cnt = m.sum((1, 2)).float()
                        vals = torch.where(m, delta, torch.full_like(delta, float("nan"))).view(B, -1)
                        med = vals.nanmedian(1).values
                        q = torch.nanquantile(vals, torch.tensor([0.1, 0.9], device=dev), dim=1)  # (2,B)
                        mean = vals.nansum(1) / cnt.clamp(min=1)
                        inr = m.view(B, -1) & (vals >= q[0][:, None]) & (vals <= q[1][:, None])
                        tmean = torch.where(inr, vals, torch.zeros_like(vals)).sum(1) / inr.sum(1).clamp(min=1)
                        mf = m.float()
                        c3 = torch.stack([(Xs * mf).sum((1, 2)), (Ys * mf).sum((1, 2)), (zb * mf).sum((1, 2))], -1) / cnt.clamp(min=1)[:, None]
                        for k, v in (("cnt", cnt), ("med", med), ("mean", mean), ("tmean", tmean), ("q10", q[0]), ("q90", q[1])):
                            res[k][:, ei, ri] = v.cpu().numpy()
                        cen[:, ei, ri] = c3.cpu().numpy()
                        if kind == "gt":
                            vv_ = vals[m.view(B, -1)].cpu().numpy()
                            hist[ei, ri] += np.histogram(vv_, bins=HIST_EDGES)[0]
                            hist_out[ei, ri] += [(vv_ < HIST_EDGES[0]).sum(), (vv_ > HIST_EDGES[-1]).sum()]
                for k in res:
                    acc[f"{kind}_{k}"].append(res[k])
                acc[f"{kind}_cen"].append(cen)

            acc["idx"].append(batch["idx"].cpu().numpy())
            acc["right"].append(right.cpu().numpy())
            acc["onscreen"].append(onscreen.cpu().numpy())
            acc["box_size"].append(bsz.cpu().numpy())
            acc["pred_j"].append(pred_j.cpu().numpy())
            acc["gt_j"].append(gt3d.cpu().numpy())
            acc["T"].append(T.cpu().numpy())
            acc["gt_recon_err"].append(((gt_j_recon - gt3d).norm(dim=-1).mean(-1) * 1000).cpu().numpy())
        if bi % 20 == 0:
            e = np.concatenate(acc["gt_recon_err"])
            print(f"batch {bi}/{len(loader)}  GT mesh recon err vs label joints: mean {e.mean():.4f} max {e.max():.4f} mm",
                  flush=True)

    R = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(os.path.join(args.out_dir, f"shard{args.shard:02d}.npz"), hist=hist, hist_out=hist_out,
                        hist_edges=HIST_EDGES, erodes=np.array(ERODES), regions=np.array(REGIONS), **R)
    print(f"saved shard {args.shard}: N={len(R['idx'])}  GT recon err max {R['gt_recon_err'].max():.4f} mm", flush=True)


if __name__ == "__main__":
    main()
