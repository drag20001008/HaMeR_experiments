"""
4가지 backbone/decoder 튜닝 전략을 비교하기 위한 학습 스크립트.

    --backbone_mode lora   : ViT backbone freeze + LoRA 어댑터만 학습
    --backbone_mode full   : ViT backbone 전체 fine-tune
    --backbone_mode freeze : ViT backbone 완전 freeze (decoder만 학습)
    --decoder_init pretrained/random : MANO transformer decoder(mano_head)를
        hamer.ckpt 사전학습 가중치에서 시작할지, 새로 초기화할지

    실험 4개:
      1. lora   + random     2. full + random
      3. freeze + random     4. freeze + pretrained

데이터: FreiHAND + HO3D_v3 + DexYCB (nas2, train_ray_hand.py와 동일 경로/스킴).
멀티 GPU: torchrun --nproc_per_node=2 로 실행하면 자동 DDP.
"""
import argparse
import os
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, open_dict

from hamer.datasets.image_dataset import ImageDataset
from hamer.models.hamer import HAMER
from hamer.models.components.lora import apply_lora_to_vit, lora_parameters


def atomic_torch_save(obj, path):
    """torch.save + os.replace using a temp file in the SAME directory as `path`.
    Avoids relying on the global TMPDIR env var (which, if pointed at a long
    nas2 path, breaks multiprocessing's AF_UNIX worker sockets) while still
    avoiding EXDEV cross-filesystem rename errors."""
    d = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def build_cfg(config_dir, use_beta_adapter=False, use_2d_head=False, kp2d_mlp_head_weight=0.01):
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="train.yaml",
                       overrides=["experiment=hamer_vit_transformer", "trainer=gpu", "data=mix_all"])
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None  # we load from hamer.ckpt ourselves
        cfg.MODEL.MANO_HEAD.BETA_ADAPTER = use_beta_adapter
        cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0             # skip discriminator entirely
        cfg.MODEL.MANO_HEAD.USE_2D_HEAD = use_2d_head  # global-token MLP 21-keypoint head
        cfg.LOSS_WEIGHTS.KP2D_MLP_HEAD = kp2d_mlp_head_weight
    return cfg


def build_model(cfg, args, is_main):
    model = HAMER(cfg, init_renderer=False)

    ckpt = torch.load(args.hamer_checkpoint, map_location="cpu")
    sd = ckpt["state_dict"]
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    mano_head_sd = {k[len("mano_head."):]: v for k, v in sd.items() if k.startswith("mano_head.")}

    # backbone always starts from the pretrained ViT regardless of tuning mode
    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=True)
    if is_main:
        print(f"[backbone] loaded pretrained weights (missing={missing}, unexpected={unexpected})")

    if args.decoder_init == "pretrained":
        missing, unexpected = model.mano_head.load_state_dict(mano_head_sd, strict=False)
        if is_main:
            print(f"[mano_head] loaded pretrained weights (missing={missing}, unexpected={unexpected})")
    elif is_main:
        print("[mano_head] random init (as constructed by HAMER.__init__)")

    # --- freeze / lora / full ---
    if args.backbone_mode == "freeze":
        model.backbone.requires_grad_(False)
    elif args.backbone_mode == "full":
        model.backbone.requires_grad_(True)
    elif args.backbone_mode == "lora":
        model.backbone.requires_grad_(False)
        n_wrapped = apply_lora_to_vit(model.backbone, r=args.lora_r, alpha=args.lora_alpha)
        if is_main:
            print(f"[lora] wrapped {n_wrapped} linear layers (r={args.lora_r}, alpha={args.lora_alpha})")
    else:
        raise ValueError(args.backbone_mode)

    model.mano_head.requires_grad_(True)  # decoder always trains in all 4 experiments
    return model


def get_trainable_params(model, args):
    params = list(model.mano_head.parameters())
    if args.backbone_mode == "full":
        params += list(model.backbone.parameters())
    elif args.backbone_mode == "lora":
        params += lora_parameters(model.backbone)
    # 'freeze' mode adds nothing from backbone
    return [p for p in params if p.requires_grad]


def build_train_dataset(cfg, args):
    dexycb_full = ImageDataset(cfg, args.dexycb_npz, args.dexycb_img_dir, train=True, rescale_factor=-1)
    train_idx = np.where(dexycb_full.data["official_split"] == "train")[0]
    parts = [Subset(dexycb_full, train_idx)]
    print(f"DexYCB train: {len(train_idx)} / {len(dexycb_full)}")

    ho3d_full = ImageDataset(cfg, args.ho3d_npz, args.ho3d_img_dir, train=True, rescale_factor=-1)
    parts.append(ho3d_full)
    print(f"HO3D (전체 사용): {len(ho3d_full)}")

    freihand_full = ImageDataset(cfg, args.freihand_npz, args.freihand_img_dir, train=True, rescale_factor=-1)
    parts.append(freihand_full)
    print(f"FreiHAND (전체 사용): {len(freihand_full)}")

    return ConcatDataset(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_mode", choices=["lora", "full", "freeze"], required=True)
    parser.add_argument("--decoder_init", choices=["random", "pretrained"], required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_2d_head", type=lambda s: s.lower() in ("true", "1", "yes"), default=False,
                         help="separate MLP head reading the global token (no extra query tokens), "
                              "outputs 21x2 keypoints via an auxiliary 2D loss")
    parser.add_argument("--kp2d_mlp_head_weight", type=float, default=0.01)
    parser.add_argument("--use_beta_adapter", type=lambda s: s.lower() in ("true", "1", "yes"), default=False,
                         help="condition the ViT context tokens on GT beta via the zero-init BetaContextAdapter")
    parser.add_argument("--hamer_checkpoint", default="_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    parser.add_argument("--config_dir", default="/workspace/drag1008/hamer/hamer/configs_hydra")
    parser.add_argument("--dexycb_npz", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz")
    parser.add_argument("--dexycb_img_dir", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--ho3d_npz", default="/nas2/lait/1000_Members/jaehoon/HO3D_v3/ho3d_annotations_v2.npz")
    parser.add_argument("--ho3d_img_dir", default="/nas2/lait/1000_Members/jaehoon/HO3D_v3")
    parser.add_argument("--freihand_npz", default="/nas2/lait/1000_Members/jaehoon/FreiHAND/freihand_annotations.npz")
    parser.add_argument("--freihand_img_dir", default="/nas2/lait/1000_Members/jaehoon/FreiHAND")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32, help="GPU 1개당 배치 크기")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every_n_steps", type=int, default=20)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"distributed={is_distributed}, world_size={world_size}, "
              f"backbone_mode={args.backbone_mode}, decoder_init={args.decoder_init}, "
              f"use_2d_head={args.use_2d_head}, use_beta_adapter={args.use_beta_adapter}")

    use_wandb = is_main and args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    cfg = build_cfg(args.config_dir, use_beta_adapter=args.use_beta_adapter,
                     use_2d_head=args.use_2d_head, kp2d_mlp_head_weight=args.kp2d_mlp_head_weight)
    model = build_model(cfg, args, is_main).to(device)

    train_dataset = build_train_dataset(cfg, args)
    sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                               sampler=sampler, num_workers=args.num_workers, drop_last=True,
                               pin_memory=True)

    trainable_params = get_trainable_params(model, args)
    n_trainable = sum(p.numel() for p in trainable_params)
    if is_main:
        print(f"trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    if args.resume_from_checkpoint:
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            print(f"resumed from {args.resume_from_checkpoint}, starting at epoch {start_epoch}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if is_distributed else model

    def move_to_device(batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = move_to_device(v)
            else:
                out[k] = v
        return out

    global_step = 0
    for epoch in range(start_epoch, args.max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_loader):
            batch = move_to_device(batch)
            output = raw_model.forward_step(batch, train=True)
            loss = raw_model.compute_loss(batch, output, train=True)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            if is_main and global_step % args.print_every_n_steps == 0:
                losses = {k: v.item() for k, v in output["losses"].items()}
                print(f"epoch {epoch} step {step} global_step {global_step} "
                      f"loss {loss.item():.4f} {losses}")
                if use_wandb:
                    wandb.log({"train/" + k: v for k, v in losses.items()}, step=global_step)
                    wandb.log({"train/loss_total": loss.item()}, step=global_step)
            global_step += 1

        if is_main:
            ckpt_path = os.path.join(args.output_dir, f"epoch={epoch:03d}.pt")
            atomic_torch_save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
