"""
WiLoR fine-tuning 스크립트 (beta conditioning 실험용).

- baseline: 기존과 동일하게 shape(beta)도 네트워크가 예측
- --beta_conditioning: GT beta를 조건으로 주입 (vit.py / refinement_net.py 수정 반영)

DexYCB를 subject 단위로 train/val 분할해서 wilor_final.ckpt에서 fine-tuning.
"""
import argparse
import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset

from wilor.models import load_wilor
from wilor.models.heads import RefineNet
from wilor.datasets.image_dataset import ImageDataset
from wilor.datasets.mocap_dataset import MoCapDataset


class PrintProgressCallback(pl.Callback):
    """rich progress bar가 non-tty 출력에서 flush 안 되는 문제를 피하기 위해
    매 N step마다 loss를 직접 print (PYTHONUNBUFFERED=1과 함께 쓰면 즉시 로그 파일에 찍힘).
    동시에 rank 0에서만 wandb에도 같은 값을 로깅한다.
    (wilor.py의 validation 로깅이 self.logger.experiment.add_scalar를 직접 호출하므로
    trainer의 메인 logger는 건드리지 않고, wandb는 이 콜백에서 별도로 직접 기록한다.)"""

    def __init__(self, every_n_steps=20, wandb_project=None, wandb_run_name=None, wandb_config=None):
        self.every_n_steps = every_n_steps
        self._last_time = None
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self._wandb = None

    def setup(self, trainer, pl_module, stage):
        if self.wandb_project and trainer.is_global_zero and self._wandb is None:
            import wandb
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=self.wandb_config)
            self._wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return
        now = time.time()
        rate = ""
        step_per_s = None
        if self._last_time is not None:
            dt = now - self._last_time
            step_per_s = self.every_n_steps / dt
            rate = f", {step_per_s:.2f} step/s"
        self._last_time = now

        losses = outputs.get("losses", {}) if isinstance(outputs, dict) else {}
        loss_str = ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
        print(
            f"[epoch {trainer.current_epoch}] step {step} "
            f"({batch_idx}/{trainer.num_training_batches}){rate} {loss_str}",
            flush=True,
        )

        if self._wandb is not None:
            log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
            log_dict["epoch"] = trainer.current_epoch
            if step_per_s is not None:
                log_dict["step_per_s"] = step_per_s
            self._wandb.log(log_dict, step=step)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f"[epoch {trainer.current_epoch}] validation done at step {trainer.global_step}", flush=True)


class JointDataset(Dataset):
    """ImageDataset + MoCapDataset을 wilor.py의 training_step이 기대하는
    {'img': ..., 'mocap': ...} 형태로 묶어주는 wrapper."""

    def __init__(self, img_dataset, mocap_dataset):
        self.img_dataset = img_dataset
        self.mocap_dataset = mocap_dataset

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img_item = self.img_dataset[idx]
        mocap_idx = np.random.randint(0, len(self.mocap_dataset))
        mocap_item = self.mocap_dataset[mocap_idx]
        return {"img": img_item, "mocap": mocap_item}


def load_model_and_cfg(checkpoint_path, model_cfg_path, beta_conditioning: bool, bbox_info: bool = False,
                        scale_loss: bool = False, scale_loss_weight: float = 0.05, real_focal: bool = False,
                        freeze_backbone: bool = False, reinit_refine_net: bool = False, lr: float = 1e-5):
    # load_wilor()가 MANO 경로 치환(${MANO.DATA_DIR} 등)을 올바르게 처리해줌
    # (주의: load_from_checkpoint(strict=False)가 refine_net도 pretrained 가중치를
    #  그대로 로드하므로, reinit_refine_net=True면 아래에서 이걸 새로 덮어써야 함)
    model, cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=model_cfg_path)
    cfg.defrost()
    if "EXPERIMENT" not in cfg:
        from yacs.config import CfgNode as CN
        cfg.EXPERIMENT = CN(new_allowed=True)
    cfg.EXPERIMENT.BETA_CONDITIONING = beta_conditioning
    cfg.EXPERIMENT.BBOX_INFO = bbox_info
    cfg.EXPERIMENT.SCALE_LOSS = scale_loss
    cfg.EXPERIMENT.SCALE_LOSS_WEIGHT = scale_loss_weight
    cfg.EXPERIMENT.REAL_FOCAL = real_focal
    cfg.EXPERIMENT.FREEZE_BACKBONE = freeze_backbone
    cfg.TRAIN.LR = lr
    cfg.freeze()
    # cfg는 model.cfg/model.backbone.cfg/model.refine_net.cfg가 공유하는 동일 객체라
    # in-place로 바꾸면 전부 반영됨

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model.backbone.eval()

    if reinit_refine_net:
        # 체크포인트에서 로드된 pretrained refine_net을 버리고 무작위 초기화된
        # 새 RefineNet으로 교체 (DexYCB로 처음부터 학습시키기 위함)
        model.refine_net = RefineNet(cfg, feat_dim=1280, upscale=3)

    return model, cfg


def split_by_subject(dataset: ImageDataset, val_subjects):
    subjects = dataset.data["subject"]
    val_mask = np.isin(subjects, val_subjects)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


def split_by_official(dataset: ImageDataset, train_split="train", val_split="val"):
    """DexYCB 공식 s0 split(dexycb_to_npz.py가 만든 'official_split' 필드) 기준 분리.
    test(78,768)는 여기서 아예 안 씀 - 최종 평가 때만 별도로 사용."""
    official = dataset.data["official_split"]
    train_idx = np.where(official == train_split)[0]
    val_idx = np.where(official == val_split)[0]
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", default="/nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations.npz")
    parser.add_argument("--img_dir", default="/nas2/lait/1000_Members/jaehoon/DexYCB")
    parser.add_argument("--model_config", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--init_checkpoint", default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta_conditioning", action="store_true")
    parser.add_argument("--bbox_info", action="store_true",
                         help="CLIFF 스타일 bbox-info(cx/f, cy/f, b/f)를 카메라 토큰에 추가 조건화")
    parser.add_argument("--scale_loss", action="store_true",
                         help="pred_cam(s,tx,ty)을 GT 절대 pelvis 위치에서 역산한 cam_gt와 직접 비교하는 loss 추가")
    parser.add_argument("--scale_loss_weight", type=float, default=0.05)
    parser.add_argument("--real_focal", action="store_true",
                         help="가짜 고정 focal length 대신 crop에 대응하는 실제 focal length(fx*IMAGE_SIZE/box_size) 사용 "
                              "- 기존 2D reprojection loss만으로 pred_cam이 절대 depth를 반영하도록 학습됨")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="ViT backbone을 완전히 고정(requires_grad=False, eval mode 유지)하고 "
                              "refine_net만 학습")
    parser.add_argument("--reinit_refine_net", action="store_true",
                         help="체크포인트에서 로드된 pretrained refine_net을 버리고 무작위 초기화된 "
                              "새 RefineNet으로 교체 후 학습 (freeze_backbone과 함께 쓰는 걸 염두에 둔 옵션)")
    parser.add_argument("--lr", type=float, default=1e-5, help="cfg.TRAIN.LR 오버라이드")
    parser.add_argument("--val_subjects", nargs="+", default=["subject-09", "subject-10"])
    parser.add_argument("--split_mode", choices=["subject", "official"], default="official",
                         help="official: DexYCB 공식 s0 split(official_split 필드) 사용, "
                              "subject: 기존처럼 --val_subjects로 subject 통째로 분리")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--print_every_n_steps", type=int, default=20)
    parser.add_argument("--wandb_project", default=None, help="설정하면 wandb 로깅 활성화")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--limit_val_batches", type=int, default=50,
                         help="중간 점검용 validation은 전체(10만+) 대신 일부만 사용 (0이면 전체)")
    parser.add_argument("--resume_from_checkpoint", default=None,
                         help="크래시 등으로 중단된 학습을 이 체크포인트(예: last.ckpt)에서 이어서 진행")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg = load_model_and_cfg(args.init_checkpoint, args.model_config, args.beta_conditioning, args.bbox_info,
                                     args.scale_loss, args.scale_loss_weight, args.real_focal,
                                     args.freeze_backbone, args.reinit_refine_net, args.lr)

    # validation_step이 매 배치마다 pyrender로 mesh를 렌더링해서 tensorboard에 올리는데,
    # headless GPU 노드에서 OpenGL 컨텍스트 생성이 멈추거나 극도로 느려서 DDP 전체가
    # NCCL timeout으로 죽는 문제가 있었음. 렌더링/로깅은 이 실험에서 필요 없으므로 비활성화.
    model.tensorboard_logging = lambda *a, **kw: None

    img_dataset_train = ImageDataset(cfg, args.dataset_npz, args.img_dir, train=True, rescale_factor=-1)
    img_dataset_val = ImageDataset(cfg, args.dataset_npz, args.img_dir, train=False, rescale_factor=-1)

    if args.split_mode == "official":
        train_idx, val_idx = split_by_official(img_dataset_train)
    else:
        train_idx, val_idx = split_by_subject(img_dataset_train, args.val_subjects)
    print(f"split_mode={args.split_mode}, train samples: {len(train_idx)}, val samples: {len(val_idx)}")

    train_img_subset = Subset(img_dataset_train, train_idx)
    val_img_subset = Subset(img_dataset_val, val_idx)

    mocap_dataset = MoCapDataset(args.dataset_npz)

    # training_step은 joint_batch['img']/['mocap']을 직접 언패킹하므로 JointDataset 필요
    train_dataset = JointDataset(train_img_subset, mocap_dataset)
    # validation_step은 batch를 그대로 forward_step에 넘기므로 순수 이미지 배치여야 함
    val_dataset = val_img_subset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        every_n_epochs=1,
        every_n_train_steps=0,       # step 기반 저장 비활성화 (명시적으로)
        save_on_train_epoch_end=True,  # val_check_interval 때문에 validation마다 저장되는 것 방지, 진짜 epoch 끝에서만 저장
        save_last=True,
        save_top_k=-1,  # 매 epoch 체크포인트를 전부 보존 (비교 실험용)
        monitor=None,   # HAMER/WiLoR validation_step은 로깅만 하므로 last만 저장
    )

    # beta_conditioning=True일 때 backbone.decshape/refine_net.dec_shape가 loss에 전혀
    # 기여하지 않아(unused parameters) DDP 기본 설정에서 에러가 나므로 find_unused_parameters 필요.
    # timeout도 넉넉하게 늘려서(기본 30분) 느린 배치 하나 때문에 전체가 죽는 걸 방지.
    if args.devices > 1:
        from datetime import timedelta
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(minutes=90))
    else:
        strategy = "auto"

    print_callback = PrintProgressCallback(
        every_n_steps=args.print_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=vars(args),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback, print_callback],
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else 1.0,
        precision="16-mixed",
        log_every_n_steps=50,
        enable_progress_bar=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
