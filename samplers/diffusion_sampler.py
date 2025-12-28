import torch
import torch.nn as nn
from torchvision.utils import save_image
from typing import Tuple
import os


class DiffusionSampler(nn.Module):
    """sampler for variance preserving sde diffusion"""
    def __init__(self, score_net: nn.Module, reverse_vp: nn.Module, output_size: Tuple[int, int],
                 batch_size: int, in_channels: int, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.score_net = score_net.to(self.device)
        self.reverse_vp = reverse_vp.to(self.device)
        self.output_size = output_size
        self.batch_size = batch_size
        self.in_channels = in_channels

    def forward(self, store_path: str) -> torch.Tensor:
        noisy_x = torch.randn(
            self.batch_size, self.in_channels, self.output_size[0], self.output_size[1]
        ).to(self.device)
        self.score_net.eval()
        self.reverse_vp.eval()
        with torch.no_grad():
            xt = noisy_x
            for t in reversed(range(self.reverse_vp.vs.num_steps)):
                noise = torch.randn_like(xt)
                time_ = torch.full((self.batch_size, ), t).to(self.device)
                t_norm = time_.float() / (self.reverse_vp.vs.num_steps - 1)
                pred_noise = self.score_net(xt, t_norm)
                xt = self.reverse_vp(xt, pred_noise, time_, noise)
            x0 = torch.clamp(xt, min=-1.0, max=1.0)
            os.makedirs(store_path, exist_ok=True)
            for i in range(x0.size(0)):
                img_path = os.path.join(store_path, f"img_{i + 1}.png")
                save_image(x0[i], img_path)
        return x0