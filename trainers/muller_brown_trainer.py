import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from typing import Callable, Dict, Optional
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os



class MBTrainer(nn.Module):
    """trainer for müller-brown experiment"""
    def __init__(
            self,
            mb_net: nn.Module,
            fwd: nn.Module,
            data_loader,
            optim: torch.optim.Optimizer,
            fp_gate: Callable,
            fp_loss: Callable, # fokker-planck loss
            dsm_loss: Callable, # denoising score matching loss
            noise_fn: Callable, # computes noise from energy
            fp_resid: Callable, # computes weak fokker-planck residuals
            epochs: int,
            device: torch.device | None = None,
            grad_acc: int = 1,
            checkpoint: int = 10,
            log_freq: int = 10,
            store_path: str = "./mb_train",
            warmup_steps: int = 100,
            fp_alpha: float = 5e-4,
            lambda_t: Callable[[torch.Tensor], torch.Tensor] = lambda t: torch.exp(-t), # time-dependent weighting λ(t)
            eps_time: float = 1e-5,
            gate_params: Optional[Dict] = None,
            *args
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mb_net = mb_net.to(self.device)
        self.fwd = fwd.to(self.device)
        self.data_loader = data_loader
        self.optim = optim
        self.fp_gate = fp_gate
        self.fp_loss = fp_loss
        self.dsm_loss = dsm_loss
        self.noise_fn = noise_fn
        self.fp_resid = fp_resid
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.lambda_t = lambda_t
        self.fp_alpha = fp_alpha
        self.eps_time = eps_time
        self.gate_params = {"k": 1.2, "snr_min": 0.5, "t_scale": 0.1, "sharpness": 5.0, "eps": 0.1}\
            if gate_params is None else gate_params
        self.global_step = 0
        self.base_lr = optim.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)
        self.losses = {'total_losses': [], 'dsm_losses': [], 'fp_losses': []}

    def forward(self):
        self.mb_net.train()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            mean_losses = self.train_batch(pbar)
            self.losses['total_losses'].append(mean_losses[0])
            self.losses['dsm_losses'].append(mean_losses[1])
            self.losses['fp_losses'].append(mean_losses[2])
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optim.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | "
                      f"Total Loss: {mean_losses[0]:.4f} | DSM Loss: {mean_losses[1]:.4f} "
                      f"| FP-Loss: {mean_losses[2]:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self.save_checkpoint(epoch + 1, mean_losses[0])
            if mean_losses[0] < self.best_loss:
                self.best_loss = mean_losses[0]
                self.save_checkpoint(epoch + 1, mean_losses[0], is_best = True)
        return self.losses

    def train_batch(self, pbar):
        """train one epoch of teacher model"""
        total_losses = []
        dsm_losses = []
        fp_losses = []
        for step, batch in enumerate(pbar):
            x = batch.to(self.device)
            step_losses = self.train_step(x)
            step_losses[0].backward()
            if (step + 1) % self.grad_acc == 0:
                clip_grad_norm_(self.mb_net.parameters(), max_norm=1.0)
                self.optim.step()
                self.optim.zero_grad()
                self.global_step += 1
                self.update_learning_rate()
            total_losses.append(step_losses[0].item() * self.grad_acc)
            pbar.set_postfix({
                'total': f'{step_losses[0].item() * self.grad_acc:.4f}',
                'dsm': f'{step_losses[1].item() * self.grad_acc:.4f}',
                'fp': f'{step_losses[2].item() * self.grad_acc:.4f}'
            })
            dsm_losses.append(step_losses[1].item() * self.grad_acc)
            fp_losses.append(step_losses[2].item() * self.grad_acc)
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        total_loss = sum(total_losses) / len(total_losses)
        dsm_loss = sum(dsm_losses) / len(dsm_losses)
        fp_loss = sum(fp_losses) / len(fp_losses)
        return total_loss, dsm_loss, fp_loss

    def train_step(self, x: torch.Tensor):
        noise = torch.randn_like(x)
        x = x.clone()
        noise = noise.clone()
        t = self.sample_time(x.shape[0], self.eps_time)
        lambda_val = self.lambda_t(t.mean())
        xt, true_score = self.fwd(x, noise, t)
        xt = xt.detach().requires_grad_(True)
        logp = self.mb_net(xt, t)
        pred_noise = self.noise_fn(logp, xt, t, self.fwd.vs)
        dsm_loss = lambda_val * self.dsm_loss(pred_noise, noise) # weighted dsm-loss
        fp_loss = torch.tensor(0.0, device=self.device)
        fp_mask, activate_fp = self.fp_gate(
            x, t, self.fwd.vs, k=self.gate_params['k'], snr_min=self.gate_params['snr_min'],
            t_scale=self.gate_params['t_scale'], sharpness=self.gate_params['sharpness'], eps=self.gate_params['eps']
        )
        if activate_fp:
            xt_active, t_active = self.subset_graph_data(xt, t, torch.unique(fp_mask))
            seed1 = self.global_step * 1000
            r1 = self.fp_resid(self.mb_net, xt_active, t_active, self.fwd.vs, seed1)
            seed2 = self.global_step * 1001
            r2 = self.fp_resid(self.mb_net, xt_active, t_active, self.fwd.vs, seed2)
            fp_lambda_val = self.lambda_t(t_active.mean())
            fp_loss = fp_lambda_val * self.fp_loss(r1, r2, alpha=self.fp_alpha) # weighted fp-loss
        total_loss = (dsm_loss + fp_loss) / self.grad_acc
        fp_loss = fp_loss / self.grad_acc
        dsm_loss = dsm_loss / self.grad_acc
        return total_loss, dsm_loss, fp_loss

    def subset_graph_data(self, xt, t, active_atom_mask):
        xt_active = xt[active_atom_mask]
        t_active = t[active_atom_mask]
        return xt_active, t_active

    def update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        checkpoint = {
            'current_epoch': epoch,
            'mb_net_state': self.mb_net.state_dict(),
            'optim_state': self.optim.state_dict(),
            'current_loss': loss,
            'losses': self.losses,
            'variance_scheduler': self.fwd.vs.state_dict(),
            'max_epochs': self.epochs,
            'base_larning_rate': self.base_lr,
            'global_step': self.global_step

        }
        filename = "mb_best.pth" if is_best else f"mb_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch} with loss {loss: .4f}")

    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)