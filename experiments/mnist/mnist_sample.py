import torch
import torch.nn as nn
from diffusion.reverse import ReverseVP
from score_nets.tiny_unet import TinyUNet
from samplers.diffusion_sampler import DiffusionSampler
from diffusion.schedules import LinearVS
from configs.load_config import load_config
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "mnist_vp_sde.yaml"
cfg = load_config(str(config_path))


vs = LinearVS(
    num_steps=cfg["diffusion"]["num_steps"],
    beta_start=cfg["diffusion"]["beta_start"],
    beta_end=cfg["diffusion"]["beta_end"],
    start=cfg["diffusion"]["time_start"],
    end=cfg["diffusion"]["time_end"],
).to("cuda")

reverse_vp = ReverseVP(vs).to("cuda")
print(sum(p.numel() for p in reverse_vp.parameters()))

checkpoint = torch.load("/home/loqman/Downloads/projs/physics_aware_diffusion/experiments/mnist/mnist_params.pth", map_location="cpu")
score_net = TinyUNet(
    in_channels=cfg["dataset"]["channels"],
    base_channels=cfg["model"]["base_channels"],
    time_dim=cfg["model"]["time_embedding_dim"],
).to("cuda")

score_net.load_state_dict(checkpoint["score_net_state"])
score_net.eval()

sampler = DiffusionSampler(
    score_net=score_net,
    reverse_vp=reverse_vp,
    output_size=(28, 28),
    batch_size=50,
    in_channels=1
)

sampler("../mnist/results")



