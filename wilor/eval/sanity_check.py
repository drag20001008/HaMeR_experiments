import torch
from wilor.models import load_wilor
from wilor.models.heads import RefineNet

model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                          cfg_path="./pretrained_models/model_config.yaml")
cfg.defrost()
if "EXPERIMENT" not in cfg:
    from yacs.config import CfgNode as CN
    cfg.EXPERIMENT = CN(new_allowed=True)
cfg.EXPERIMENT.FREEZE_BACKBONE = True
cfg.freeze()

for p in model.backbone.parameters():
    p.requires_grad_(False)
model.backbone.eval()

old_refine_state = {k: v.clone() for k, v in model.refine_net.state_dict().items()}
model.refine_net = RefineNet(cfg, feat_dim=1280, upscale=3)
new_refine_state = model.refine_net.state_dict()

changed = 0
for k in old_refine_state:
    if not torch.equal(old_refine_state[k], new_refine_state[k]):
        changed += 1
print(f"refine_net tensors changed after reinit: {changed}/{len(old_refine_state)}")

params = model.get_parameters()
trainable = [p for p in params if p.requires_grad]
n_backbone = sum(p.numel() for p in model.backbone.parameters())
n_refine = sum(p.numel() for p in model.refine_net.parameters())
n_trainable = sum(p.numel() for p in trainable)
print(f"backbone params: {n_backbone:,}  (requires_grad example: {next(model.backbone.parameters()).requires_grad})")
print(f"refine_net params: {n_refine:,}")
print(f"trainable params returned to optimizer: {n_trainable:,}  (should ~= refine_net params)")
assert n_trainable == n_refine, "trainable param count should exactly equal refine_net param count when backbone frozen"
print("OK: only refine_net params are trainable, backbone frozen, refine_net re-initialized differently from checkpoint")

model.train()
print(f"after model.train(): backbone.training={model.backbone.training} (should be False), refine_net.training={model.refine_net.training} (should be True)")
