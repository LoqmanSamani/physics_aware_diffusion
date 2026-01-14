import torch
import torch.nn as nn
from torchvision.utils import save_image
from diffusion.schedules import LinearVS
from diffusion.reverse import ReverseVP
from typing import Tuple
import os
from tqdm import tqdm


class MNISTSampler(nn.Module):
    """sampler for variance preserving sde diffusion"""
    def __init__(self, score_net: nn.Module, variance_scheduler: LinearVS,
                 output_size: Tuple[int, int], batch_size: int, in_channels: int,
                 device: str = "cuda", eps: float = 1e-3) -> None:
        super().__init__()
        self.device = device
        self.score_net = score_net.to(self.device)
        self.vs = variance_scheduler
        self.reverse_vp = ReverseVP(variance_scheduler, eps=eps).to(self.device)
        self.output_size = output_size
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.eps = eps

    def forward(self, num_steps: int, store_path: str) -> torch.Tensor:
        """
        sample using reverse-time sde
        arguments:
            num_steps: number of discretization steps
            store_path: path to save generated images
        """
        xt = torch.randn(
            self.batch_size, self.in_channels,
            self.output_size[0], self.output_size[1]
        ).to(self.device)
        self.score_net.eval()
        self.reverse_vp.eval()
        t_schedule = torch.linspace(1.0 - self.eps, self.eps, num_steps + 1)
        dt = torch.tensor(-1.0 * (1.0 - 2 * self.eps) / num_steps) # negative for reverse time
        iterator = tqdm(range(num_steps), desc="Sampling")
        with torch.no_grad():
            for step in iterator:
                t_current = float(t_schedule[step])
                t_batch = torch.full((self.batch_size,), t_current, dtype=torch.float32, device=self.device)
                pred_score = self.score_net(xt, t_batch)
                if step < num_steps - 1:
                    noise = torch.randn_like(xt)
                else:
                    noise = torch.zeros_like(xt)
                # take reverse sde step
                xt = self.reverse_vp(xt, pred_score, t_batch, dt, noise)
                # xt = torch.clamp(xt, min=-3.0, max=3.0)
        x0 = torch.clamp(xt, min=-1.0, max=1.0)
        os.makedirs(store_path, exist_ok=True)
        for i in range(x0.size(0)):
            img_path = os.path.join(store_path, f"img_{i + 1}.png")
            save_image((x0[i] + 1) / 2, img_path)
        return x0