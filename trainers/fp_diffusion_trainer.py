import torch
import torch.nn as nn
from typing import Callable, Optional


class FPEnergyTrainer(nn.Module):
    def __init__(
            self,
            energy_net: nn.Module,
            fp_gate: nn.Module,
            forward_vp: nn.Module,
            data_loader,
            optimizer: torch.optim.Optimizer,
            fp_loss: Callable,
            dsm_loss: Callable,
            score_fn: Callable,
            fp_residual: Callable,
            epochs: int,
            device: torch.device | None = None,
            grad_acc: int = 1,
            checkpoint: int = 5,
            log_freq: int = 1,
            store_path: str = "./checkpoints",
            warmup_steps: int = 0,
            rotation_augmentation: bool = False,
            *args
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.energy_net = energy_net.to(self.device)
        self.fp_gate = fp_gate.to(self.device)
        self.forward_vp = forward_vp.to(self.device)



    def forward(self):
        pass
