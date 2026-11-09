import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import hydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, open_dict
import torch

with initialize_config_dir(config_dir="/workspace/drag1008/hamer/hamer/configs_hydra", version_base="1.2"):
    cfg = compose(config_name="train.yaml", overrides=["experiment=hamer_vit_transformer", "trainer=cpu", "data=mix_all"])

OmegaConf.set_struct(cfg, False)
with open_dict(cfg):
    # skip the vitpose-only backbone init load; we'll load the full hamer.ckpt instead
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None

print("BETA_ADAPTER cfg:", cfg.MODEL.MANO_HEAD.get("BETA_ADAPTER"), cfg.MODEL.MANO_HEAD.get("BETA_ADAPTER_ONLY"))

from hamer.models.hamer import HAMER
model = HAMER(cfg, init_renderer=False)
model.eval()

n_adapter = sum(p.numel() for p in model.mano_head.beta_adapter.parameters())
print("adapter param count:", n_adapter)

opt_params = model.get_parameters()
n_opt = sum(p.numel() for p in opt_params)
print("params handed to optimizer:", n_opt, "== adapter only?", n_opt == n_adapter)

# --- Load real pretrained weights from hamer.ckpt ---
ckpt = torch.load("_DATA/hamer_ckpts/checkpoints/hamer.ckpt", map_location="cpu")
sd = ckpt["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing keys (expected: beta_adapter.* only):", missing)
print("unexpected keys:", unexpected)

# adapter should still be identity (zero-init) since it's not in the checkpoint
for n, p in model.mano_head.beta_adapter.named_parameters():
    if "proj_z" in n or ("up." in n and "up.weight" in n or "up.bias" in n):
        assert torch.all(p == 0), f"{n} not zero after loading ckpt!"
print("adapter still zero-init after ckpt load: OK")

# Fabricate a minimal batch matching forward_step's expectations
B = 2
batch = {
    "img": torch.randn(B, 3, 256, 256),
    "mano_params": {
        "betas": torch.randn(B, 10),
        "hand_pose": torch.zeros(B, 15, 3, 3),
        "global_orient": torch.zeros(B, 1, 3, 3),
    },
    "has_mano_params": {
        "betas": torch.tensor([1, 0]),
        "hand_pose": torch.tensor([1, 1]),
        "global_orient": torch.tensor([1, 1]),
    },
}

with torch.no_grad():
    out_train = model.forward_step(batch, train=True)
    out_eval = model.forward_step(batch, train=False)

print("pred_vertices shape:", out_train["pred_vertices"].shape)
print("pred_mano_params betas shape:", out_train["pred_mano_params"]["betas"].shape)

# with zero-init adapter, train (beta-conditioned) and eval (mean-shape-conditioned)
# outputs should currently be IDENTICAL since the adapter is still identity
diff = (out_train["pred_vertices"] - out_eval["pred_vertices"]).abs().max().item()
print("train vs eval vertex diff (should be 0.0 pre-finetune):", diff)

print("SMOKE TEST OK")
