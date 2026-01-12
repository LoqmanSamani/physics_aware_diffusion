import torch
import torch.optim as optim
from losses.dsm_losses import mse_loss
from data_loader import get_mnist_subset_dataloader
from score_nets.tiny_unet import TinyUNet
from trainers.mnist_trainer import MNISTTrainer
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from configs.load_config import load_config
import random
import numpy as np
from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "mnist_vp_sde.yaml"
cfg = load_config(str(config_path))
set_seed(cfg["experiment"]["seed"])

vs = LinearVS(
    beta_start=cfg["diffusion"]["beta_start"],
    beta_end=cfg["diffusion"]["beta_end"],
    min_variance = cfg['diffusion']['min_variance']
).to(cfg["experiment"]["device"])

forward_vp = ForwardVP(vs).to(cfg["experiment"]["device"])
#checkpoint = torch.load("/home/loqman/Downloads/projs/physics_aware_diffusion/experiments/mnist/vp_best.pth", map_location="cpu")
score_net = TinyUNet(
    in_channels=cfg["dataset"]["channels"],
    base_channels=cfg["model"]["base_channels"],
    time_dim=cfg["model"]["time_embedding_dim"],
).to(cfg["experiment"]["device"])
#score_net.load_state_dict(checkpoint['score_net_state'])
optimizer = optim.AdamW(
    score_net.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
)

data_loader = get_mnist_subset_dataloader(
    batch_size=cfg["training"]["batch_size"],
    subset_fraction=cfg["dataset"]["subset_fraction"],
)

checkpoint = torch.load("/experiments/toy_2d/mnist/vp_best.pth", map_location="cpu")
score_net.load_state_dict(checkpoint["score_net_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
vs.load_state_dict(checkpoint["scheduler"])


trainer = MNISTTrainer(
    score_net=score_net,
    forward_vp=forward_vp,
    data_loader=data_loader,
    optimizer=optimizer,
    loss_fn=mse_loss,
    epochs=cfg["training"]["epochs"],
    grad_acc=cfg["training"]["grad_accumulation"],
    checkpoint=cfg["logging"]["checkpoint_freq"],
    log_freq=cfg["logging"]["log_freq"],
    store_path=cfg["logging"]["output_dir"],
    device=cfg["experiment"]["device"],
    warmup_steps=cfg["training"]["warmup_steps"]
)

train_losses = trainer()