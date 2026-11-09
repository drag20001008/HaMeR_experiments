"""
train_backbone_experiments.py로 학습한 체크포인트를 FreiHAND / HO3D_v3 / DexYCB(test split)
세 데이터셋에 대해 평가. hamer_prev/eval_hamer_baseline.py의 GT/지표 계산 방식(CS-MPJPE, PA-MPJPE)을
그대로 재사용하되, load_hamer() 대신 우리 실험 체크포인트를 로드하도록 바꿈.
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
    def __init__(self, image_dataset):
        self.ds = image_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        kp3d_abs = item['keypoints_3d'][:, :3].astype(np.float32)
        betas = item['mano_params']['betas'].astype(np.float32)
        has_betas = np.float32(item['has_mano_params']['betas'])
        return {
            'img': torch.from_numpy(item['img']).float() if isinstance(item['img'], np.ndarray) else item['img'],
            'gt_keypoints_3d_abs': torch.from_numpy(kp3d_abs),
            'betas': torch.from_numpy(betas),
            'has_betas': torch.tensor(has_betas),
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
    dataset = HamerEvalWrapper(image_dataset)
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


def evaluate(model, dataset_name, model_cfg, device, batch_size, num_workers):
    dataset = build_dataset(dataset_name, model_cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pred_list, gt_abs_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {dataset_name}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # train=True only to unlock the GT-beta conditioning path in forward_step;
            # model.eval() above already froze BN/dropout, so this doesn't change those.
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
            pred_cam = (kp3d_rel + pred_cam_t.unsqueeze(1)).cpu().numpy()
            pred_list.append(pred_cam)
            gt_abs_list.append(batch["gt_keypoints_3d_abs"].cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    gts_abs = np.concatenate(gt_abs_list, axis=0)

    pa_dists, raw_dists = [], []
    for i in range(len(preds)):
        pred, gt = preds[i], gts_abs[i]
        raw_dists.append(np.linalg.norm(pred - gt, axis=1))
        pred_aligned = align_w_scale(gt, pred)
        pa_dists.append(np.linalg.norm(pred_aligned - gt, axis=1))

    raw_dists = np.array(raw_dists) * 1000.0
    pa_dists = np.array(pa_dists) * 1000.0
    result = {
        "dataset": dataset_name, "N": int(len(preds)),
        "cs_mpjpe_mean_mm": float(raw_dists.mean()), "cs_mpjpe_median_mm": float(np.median(raw_dists)),
        "pa_mpjpe_mean_mm": float(pa_dists.mean()), "pa_mpjpe_median_mm": float(np.median(pa_dists)),
    }
    print(f"[{dataset_name}] N={result['N']} "
          f"CS-MPJPE={result['cs_mpjpe_mean_mm']:.2f}/{result['cs_mpjpe_median_mm']:.2f}mm "
          f"PA-MPJPE={result['pa_mpjpe_mean_mm']:.2f}/{result['pa_mpjpe_median_mm']:.2f}mm")
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
        r = evaluate(model, name, cfg, device, args.batch_size, args.num_workers)
        r["tag"] = args.tag
        r["checkpoint"] = args.checkpoint
        results.append(r)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {args.out_json}")


if __name__ == "__main__":
    main()
