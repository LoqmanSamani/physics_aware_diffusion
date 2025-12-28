import torch
import torch.optim as optim
from data_loader import get_mnist_subset_dataloader
from models.score.tiny_unet import TinyUNet
from trainers.diffusion_trainer import DiffusionTrainer
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


def vp_noise_loss(pred_noise, true_noise, *args):
    return torch.mean((pred_noise - true_noise) ** 2)


project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "mnist_vp_sde.yaml"
cfg = load_config(str(config_path))
set_seed(cfg["experiment"]["seed"])

vs = LinearVS(
    num_steps=cfg["diffusion"]["num_steps"],
    beta_start=cfg["diffusion"]["beta_start"],
    beta_end=cfg["diffusion"]["beta_end"],
    start=cfg["diffusion"]["time_start"],
    end=cfg["diffusion"]["time_end"],
).to(cfg["experiment"]["device"])

forward_vp = ForwardVP(vs).to(cfg["experiment"]["device"])

score_net = TinyUNet(
    in_channels=cfg["dataset"]["channels"],
    base_channels=cfg["model"]["base_channels"],
    time_dim=cfg["model"]["time_embedding_dim"],
).to(cfg["experiment"]["device"])

optimizer = optim.AdamW(
    score_net.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
)

data_loader = get_mnist_subset_dataloader(
    batch_size=cfg["training"]["batch_size"],
    subset_fraction=cfg["dataset"]["subset_fraction"],
)




trainer = DiffusionTrainer(
    score_net=score_net,
    forward_vp=forward_vp,
    data_loader=data_loader,
    optimizer=optimizer,
    loss_fn=vp_noise_loss,
    epochs=cfg["training"]["epochs"],
    grad_acc=cfg["training"]["grad_accumulation"],
    checkpoint=cfg["logging"]["checkpoint_freq"],
    log_freq=cfg["logging"]["log_freq"],
    store_path=cfg["logging"]["output_dir"],
    device=cfg["experiment"]["device"]
)

train_losses = trainer()