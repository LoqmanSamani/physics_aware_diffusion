import torch
import torch.nn as nn
from diffusion.reverse import ReverseVP
from score_nets.tiny_unet import TinyUNet
from samplers.mnist_sampler import MNISTSampler
from diffusion.schedules import LinearVS
from configs.load_config import load_config
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "mnist_vp_sde.yaml"
cfg = load_config(str(config_path))


vs = LinearVS(
    beta_start=cfg["diffusion"]["beta_start"],
    beta_end=cfg["diffusion"]["beta_end"],
    min_variance=cfg['diffusion']['min_variance']
).to("cuda")

reverse_vp = ReverseVP(vs).to("cuda")
checkpoint = torch.load("/home/loqman/Downloads/projs/physics_aware_diffusion/experiments/mnist/vp_best.pth", map_location="cpu")
score_net = TinyUNet(
    in_channels=cfg["dataset"]["channels"],
    base_channels=cfg["model"]["base_channels"],
    time_dim=cfg["model"]["time_embedding_dim"],
).to("cuda")

score_net.load_state_dict(checkpoint["score_net_state"])
score_net.eval()
sampler = MNISTSampler(
    score_net=score_net,
    variance_scheduler=vs,
    output_size=(28, 28),
    batch_size=50,
    in_channels=1,
    device='cuda'
)
sampler(1000, "../mnist/results1")


