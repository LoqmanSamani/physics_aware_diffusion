import torch
import torch.nn as nn
from diffusion.reverse import ReverseVP, ReverseDDPM, ReverseDDPMSimplified
from score_nets.tiny_unet import TinyUNet
from samplers.mnist_samplers import MNISTSampler, MNISTSamplerDDPM
from diffusion.schedules import LinearVS
from configs.load_config import load_config
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
config_path = "/home/loqman/Downloads/projs/physics_aware_diffusion/configs/mnist_vp_sde.yaml"
cfg = load_config(str(config_path))


vs = LinearVS(
    beta_min=cfg["diffusion"]["beta_min"],
    beta_max=cfg["diffusion"]["beta_max"]
).to("cuda")

reverse_vp = ReverseVP(vs).to("cuda")
reverse_vp_ddpm = ReverseDDPM(vs).to("cuda")
reverse_vp_ddpm_s = ReverseDDPMSimplified(vs).to("cuda")

checkpoint = torch.load("/home/loqman/Downloads/projs/physics_aware_diffusion/experiments/toy_2d/mnist/vp_best.pth", map_location="cpu")
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
sampler(num_steps = 1000, store_path = "../mnist/results1", mode = 'sde', normalize = True)


ddpm_sampler = MNISTSamplerDDPM(
    score_net = score_net,
    reverse_ddpm = reverse_vp_ddpm_s, # reverse_vp_ddpm
    output_size = (28, 28),
    batch_size = 50,
    in_channels = 1,
    device = "cuda",
    eps =  1e-5
)

x0 = ddpm_sampler(1000, "../mnist/results2", False)
