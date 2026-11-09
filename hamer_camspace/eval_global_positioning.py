"""
기존 eval_backbone_experiments.py의 weak-perspective pred_cam_t 대신,
root-relative 3D pose + 2D keypoint(GT 또는 exp5의 실제 예측 pred_kp2d_mlp_head) +
원본 카메라 K로 translation T를 선형 최소자승(closed-form)으로 직접 풀어서
CS-MPJPE를 재계산한다.

3D(root-relative pose)는 crop과 무관한 metric 좌표이므로, 2D keypoint와 K도
crop 좌표계가 아니라 원본 이미지 좌표계를 써야 한다.

- GT 2D (npz의 orig_keypoints_2d, 이미 원본 좌표계) : 그대로 원본 K와 짝지어 사용 (oracle)
- 예측 2D (pred_kp2d_mlp_head, crop-local 정규화: norm = pixel_in_crop/224 - 0.5) :
    eval 시 rot=0, aug-scale=1이라 crop 변환이 단순 similarity transform이므로
        orig_x = box_center_x + norm_x * box_size
        orig_y = box_center_y + norm_y * box_size
  로 원본 좌표계로 역변환한 뒤 원본 K와 짝지어 사용 (use_2d_head=True인 실험만 가능, 즉 exp5)

세 가지(old weak-persp / new-GT2D-oracle / new-pred2d)를 나란히 출력해서 비교한다.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, open_dict

from hamer.datasets.image_dataset import ImageDataset
from hamer.models.hamer import HAMER
from hamer.models.components.lora import apply_lora_to_vit

DATASETS = {
    "freihand": dict(
        npz="/nas2/lait/1000_Members/jaehoon/FreiHAND/freihand_eval_annotations.npz",
        img_dir="/nas2/lait/1000_Members/jaehoon/FreiHAND",
        split_field=None, split=None,
    ),
    "dexycb": dict(
        npz="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz",
        img_dir="/nas2/lait/1000_Members/jaehoon/DexYCB",
        split_field="official_split", split="test",
    ),
    "ho3d": dict(
        npz="/nas2/lait/1000_Members/jaehoon/HO3D_v3/ho3d_evaluation_annotations.npz",
        img_dir="/nas2/lait/1000_Members/jaehoon/HO3D_v3",
        split_field=None, split=None,
    ),
}


class HamerEvalWrapper(Dataset):
    def __init__(self, image_dataset, npz_path):
        self.ds = image_dataset
        raw = np.load(npz_path, allow_pickle=True)
        self.fx = raw["fx"].astype(np.float32)
        self.fy = raw["fy"].astype(np.float32)
        self.px = raw["px"].astype(np.float32)
        self.py = raw["py"].astype(np.float32)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        kp3d_abs = item['keypoints_3d'][:, :3].astype(np.float32)
        betas = item['mano_params']['betas'].astype(np.float32)
        has_betas = np.float32(item['has_mano_params']['betas'])
        orig_kp2d = item['orig_keypoints_2d'][:, :2].astype(np.float32)  # (21,2) 원본 이미지 좌표 GT
        box_center = np.asarray(item['box_center'], dtype=np.float32)   # (2,)
        box_size = np.float32(item['box_size'])                          # scalar
        right = np.float32(item['right'])  # 1=오른손, 0=왼손 (get_example가 왼손을 flip해서 넣음)
        return {
            'img': torch.from_numpy(item['img']).float() if isinstance(item['img'], np.ndarray) else item['img'],
            'gt_keypoints_3d_abs': torch.from_numpy(kp3d_abs),
            'betas': torch.from_numpy(betas),
            'has_betas': torch.tensor(has_betas),
            'orig_kp2d': torch.from_numpy(orig_kp2d),
            'right': torch.tensor(right),
            'box_center': torch.from_numpy(box_center),
            'box_size': torch.tensor(box_size),
            'fx': torch.tensor(self.fx[idx]),
            'fy': torch.tensor(self.fy[idx]),
            'px': torch.tensor(self.px[idx]),
            'py': torch.tensor(self.py[idx]),
        }


def align_w_scale(mtx1, mtx2):
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    return mtx2_t


def build_dataset(name, model_cfg):
    spec = DATASETS[name]
    image_dataset = ImageDataset(model_cfg, spec["npz"], spec["img_dir"], train=False, rescale_factor=-1)
    dataset = HamerEvalWrapper(image_dataset, spec["npz"])
    if spec["split_field"] is not None:
        raw = np.load(spec["npz"], allow_pickle=True)
        split_values = raw[spec["split_field"]]
        indices = [i for i, v in enumerate(split_values) if v == spec["split"]]
        print(f"filtered {name} split='{spec['split']}': {len(indices)}/{len(split_values)} samples")
        dataset = Subset(dataset, indices)
    else:
        print(f"{name}: {len(dataset)} samples (eval file, no split filter)")
    return dataset


def build_cfg(config_dir, use_beta_adapter=False, use_2d_head=False):
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="train.yaml",
                       overrides=["experiment=hamer_vit_transformer", "trainer=gpu", "data=mix_all"])
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
        cfg.MODEL.MANO_HEAD.BETA_ADAPTER = use_beta_adapter
        cfg.MODEL.MANO_HEAD.USE_2D_HEAD = use_2d_head
        cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    return cfg


def build_model(cfg, backbone_mode, lora_r, lora_alpha, checkpoint_path, device):
    model = HAMER(cfg, init_renderer=False)
    if backbone_mode == "lora":
        n_wrapped = apply_lora_to_vit(model.backbone, r=lora_r, alpha=lora_alpha)
        print(f"[lora] wrapped {n_wrapped} linear layers (r={lora_r}, alpha={lora_alpha})")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    print(f"loaded checkpoint {checkpoint_path} (epoch={ckpt.get('epoch')}) "
          f"missing={missing} unexpected={unexpected}")

    model = model.to(device)
    model.eval()
    return model


def unmirror_left_hand(x, right):
    """
    get_example가 왼손(right==0) 이미지를 좌우 flip해서 네트워크에 넣기 때문에(HaMeR
    'if it's a left hand, we flip'), 네트워크가 내놓는 모든 3D/2D 예측은 왼손 샘플에서
    '거울에 비친' 좌표계로 나온다. 실제 카메라 좌표계(원본 2D keypoint, 실제 depth/K)와
    맞추려면 x좌표 부호를 되돌려야 한다 (fliplr_keypoints가 x <- -x로 뒤집는 것의 역연산).
    x: (..., N, D) 마지막에서 두번째 축이 keypoint/vertex, 마지막 축의 0번째가 x좌표.
    right: (B,) 1=오른손(그대로), 0=왼손(x부호 반전).
    """
    sign = (2 * right - 1)  # (B,) -> +1 or -1
    while sign.dim() < x.dim():
        sign = sign.unsqueeze(-1)
    out = x.clone()
    out[..., 0:1] = out[..., 0:1] * sign
    return out


def solve_translation(kp3d_rel, kp2d, fx, fy, px, py):
    """
    kp3d_rel: (B, 21, 3) root-relative 3D (모델 예측, crop-independent)
    kp2d:     (B, 21, 2) 원본 이미지 좌표 2D (GT 또는 예측→역변환된 값)
    fx,fy,px,py: (B,) 원본 카메라 K
    반환: T (B, 3) - closed-form 선형 최소자승 해
    """
    B, N, _ = kp3d_rel.shape
    Zx, Zy, Zz = kp3d_rel[..., 0], kp3d_rel[..., 1], kp3d_rel[..., 2]  # (B,N)
    u, v = kp2d[..., 0], kp2d[..., 1]  # (B,N)

    fx_ = fx.view(B, 1)
    fy_ = fy.view(B, 1)
    px_ = px.view(B, 1)
    py_ = py.view(B, 1)

    A_u = torch.stack([
        -fx_.expand(B, N), torch.zeros(B, N, device=kp3d_rel.device), (u - px_)
    ], dim=-1)  # (B,N,3)
    b_u = fx_ * Zx + px_ * Zz - u * Zz  # (B,N)

    A_v = torch.stack([
        torch.zeros(B, N, device=kp3d_rel.device), -fy_.expand(B, N), (v - py_)
    ], dim=-1)  # (B,N,3)
    b_v = fy_ * Zy + py_ * Zz - v * Zz  # (B,N)

    A = torch.cat([A_u, A_v], dim=1)  # (B, 2N, 3)
    b = torch.cat([b_u, b_v], dim=1).unsqueeze(-1)  # (B, 2N, 1)

    sol = torch.linalg.lstsq(A, b).solution  # (B, 3, 1)
    return sol.squeeze(-1)  # (B, 3)


def evaluate(model, dataset_name, model_cfg, device, batch_size, num_workers, use_2d_head):
    dataset = build_dataset(dataset_name, model_cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pred_old_list, pred_gt2d_list, pred_pred2d_list, pred_rootgt_list, gt_abs_list = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {dataset_name}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_batch = {
                'img': batch['img'],
                'mano_params': {'betas': batch['betas']},
                'has_mano_params': {'betas': batch['has_betas']},
            }
            out = model.forward_step(model_batch, train=True)
            pred_keypoints_3d = out['pred_keypoints_3d']
            pred_cam_t = out['pred_cam_t']
            root = pred_keypoints_3d[:, 0:1, :]
            kp3d_rel = pred_keypoints_3d - root

            # (1) 기존 방식: weak-perspective pred_cam_t
            pred_old = (kp3d_rel + pred_cam_t.unsqueeze(1)).cpu().numpy()

            # (2) GT 2D(oracle) + 원본 K로 closed-form translation solve
            T_gt2d = solve_translation(kp3d_rel, batch['orig_kp2d'], batch['fx'], batch['fy'], batch['px'], batch['py'])
            pred_gt2d = (kp3d_rel + T_gt2d.unsqueeze(1)).cpu().numpy()

            # (4) translation을 그냥 GT wrist(root) 절대좌표로 고정 -> root-relative MPJPE와 동치
            T_rootgt = batch['gt_keypoints_3d_abs'][:, 0, :]
            pred_rootgt = (kp3d_rel + T_rootgt.unsqueeze(1)).cpu().numpy()

            pred_old_list.append(pred_old)
            pred_gt2d_list.append(pred_gt2d)
            pred_rootgt_list.append(pred_rootgt)
            gt_abs_list.append(batch["gt_keypoints_3d_abs"].cpu().numpy())

            # (3) 모델이 실제로 예측한 2D(pred_kp2d_mlp_head, crop-local normalized) -> 원본 좌표로 역변환 -> solve
            if use_2d_head and 'pred_kp2d_mlp_head' in out:
                B = kp3d_rel.shape[0]
                pred_kp2d_norm = out['pred_kp2d_mlp_head'].view(B, 21, 2)  # crop-local normalized (pixel/224 - 0.5)
                box_center = batch['box_center'].view(B, 1, 2)
                box_size = batch['box_size'].view(B, 1, 1)
                pred_kp2d_orig = box_center + pred_kp2d_norm * box_size  # 원본 이미지 좌표로 역변환
                T_pred2d = solve_translation(kp3d_rel, pred_kp2d_orig, batch['fx'], batch['fy'], batch['px'], batch['py'])
                pred_pred2d = (kp3d_rel + T_pred2d.unsqueeze(1)).cpu().numpy()
                pred_pred2d_list.append(pred_pred2d)

    preds_old = np.concatenate(pred_old_list, axis=0)
    preds_gt2d = np.concatenate(pred_gt2d_list, axis=0)
    preds_rootgt = np.concatenate(pred_rootgt_list, axis=0)
    gts_abs = np.concatenate(gt_abs_list, axis=0)

    def cs_mpjpe(preds, gts):
        raw_dists = np.linalg.norm(preds - gts, axis=2) * 1000.0  # (N,21) in mm
        return float(raw_dists.mean()), float(np.median(raw_dists))

    old_mean, old_median = cs_mpjpe(preds_old, gts_abs)
    gt2d_mean, gt2d_median = cs_mpjpe(preds_gt2d, gts_abs)
    rootgt_mean, rootgt_median = cs_mpjpe(preds_rootgt, gts_abs)

    result = {
        "dataset": dataset_name, "N": int(len(preds_old)),
        "cs_mpjpe_old_mean_mm": old_mean, "cs_mpjpe_old_median_mm": old_median,
        "cs_mpjpe_gt2d_mean_mm": gt2d_mean, "cs_mpjpe_gt2d_median_mm": gt2d_median,
        "cs_mpjpe_rootgt_mean_mm": rootgt_mean, "cs_mpjpe_rootgt_median_mm": rootgt_median,
    }
    msg = (f"[{dataset_name}] N={result['N']} "
           f"CS-MPJPE(old,weak-persp)={old_mean:.2f}/{old_median:.2f}mm  "
           f"CS-MPJPE(GT-2D+K oracle)={gt2d_mean:.2f}/{gt2d_median:.2f}mm  "
           f"CS-MPJPE(root=GT wrist)={rootgt_mean:.2f}/{rootgt_median:.2f}mm")

    if pred_pred2d_list:
        preds_pred2d = np.concatenate(pred_pred2d_list, axis=0)
        pred2d_mean, pred2d_median = cs_mpjpe(preds_pred2d, gts_abs)
        result["cs_mpjpe_pred2d_mean_mm"] = pred2d_mean
        result["cs_mpjpe_pred2d_median_mm"] = pred2d_median
        msg += f"  CS-MPJPE(pred-2D+K)={pred2d_mean:.2f}/{pred2d_median:.2f}mm"

    print(msg)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_mode", choices=["lora", "full", "freeze"], required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use_beta_adapter", type=lambda s: s.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--use_2d_head", type=lambda s: s.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, tag={args.tag}, checkpoint={args.checkpoint}, "
          f"use_beta_adapter={args.use_beta_adapter}, use_2d_head={args.use_2d_head}")

    cfg = build_cfg(args.config_dir, use_beta_adapter=args.use_beta_adapter, use_2d_head=args.use_2d_head)
    model = build_model(cfg, args.backbone_mode, args.lora_r, args.lora_alpha, args.checkpoint, device)

    results = []
    for name in ["freihand", "ho3d", "dexycb"]:
        r = evaluate(model, name, cfg, device, args.batch_size, args.num_workers, args.use_2d_head)
        r["tag"] = args.tag
        r["checkpoint"] = args.checkpoint
        results.append(r)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {args.out_json}")


if __name__ == "__main__":
    main()
