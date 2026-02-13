import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Callable, Dict, Optional
from torch.nn.utils import clip_grad_norm_
from contextlib import contextmanager
from tqdm import tqdm
import os


class MBTrainer(nn.Module):
    """trainer (trainable gate) for müller-brown experiment"""
    def __init__(
            self,
            mb_net: nn.Module,
            fwd: nn.Module,
            data_loader,
            optim: torch.optim.Optimizer,
            fp_gate: nn.Module,
            fp_loss: Callable,
            noise_fn: Callable,
            score_fn: Callable,
            fp_resid: Callable,
            epochs: int,
            device: torch.device | None = None,
            grad_acc: int = 1,
            checkpoint: int = 10,
            log_freq: int = 10,
            store_path: str = "./mb_train",
            warmup_steps: int = 100,
            fp_alpha: float = 5e-4,
            lambda_t: Callable[[torch.Tensor], torch.Tensor] = lambda t: torch.exp(-t),
            eps_time: float = 1e-5,
            gate_epochs: int = 1,  # number of training epochs in each fp-gate fine-tune process
            gate_optim: Optional[torch.optim.Optimizer] = None,
            gate_loss: Optional[torch.nn.functional] = None,
            fp_q: float = 0.9,  # top 10 percent get fokker-planck regularization
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
        self.noise_fn = noise_fn
        self.score_fn = score_fn
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
        self.gate_epochs = gate_epochs
        self.gate_optim = gate_optim or torch.optim.Adam(self.fp_gate.parameters(), lr=1e-4)
        self.gate_loss = gate_loss or torch.nn.functional.binary_cross_entropy
        self.fp_q = fp_q
        self.init_epoch = 0
        self.global_step = 0
        self.base_lr = optim.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)
        self.losses = {'total_losses': [], 'dsm_losses': [], 'fp_losses': [], 'gate_losses': []}

    def forward(self):
        self.mb_net.train()
        self.fp_gate.eval()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            if epoch < 10:
                mean_losses = self.train_batch(pbar, use_fp = False)
            else:
                mean_losses = self.train_batch(pbar, use_fp = True)
            self.losses['total_losses'].append(mean_losses[0])
            self.losses['dsm_losses'].append(mean_losses[1])
            self.losses['fp_losses'].append(mean_losses[2])
            if epoch % 10 == 0 and epoch != 0:
                print('--' * 50)
                print('\nFine-Tuning FP-Gate')
                print('--' * 50)
                for _ in range(self.gate_epochs):
                    pbar_ = tqdm(self.data_loader)
                    gate_loss = self.gate_batch(pbar_)
                    self.losses['gate_losses'].append(gate_loss)
                print()
                print('--' * 50)
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optim.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | "
                      f"Total Loss: {mean_losses[0]:.4f} | DSM Loss: {mean_losses[1]:.4f} "
                      f"| FP-Loss: {mean_losses[2]:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self.save_checkpoint(epoch + 1, mean_losses[0])
            if mean_losses[0] < self.best_loss:
                self.best_loss = mean_losses[0]
                self.save_checkpoint(epoch + 1, mean_losses[0], is_best=True)

        return self.losses

    def train_batch(self, pbar, use_fp = True):
        """train one epoch"""
        total_losses = []
        dsm_losses = []
        fp_losses = []
        for step, batch in enumerate(pbar):
            x = batch.to(self.device)
            step_losses = self.train_step(x, use_fp)
            step_losses[0].backward()
            if (step + 1) % self.grad_acc == 0:
                clip_grad_norm_(self.mb_net.parameters(), max_norm=1.0)
                self.optim.step()
                self.optim.zero_grad()
                self.global_step += 1
                self.update_learning_rate()
            total_losses.append(step_losses[0].item() * self.grad_acc)
            dsm_losses.append(step_losses[1].item() * self.grad_acc)
            fp_losses.append(step_losses[2].item() * self.grad_acc)
            pbar.set_postfix({
                'total': f'{step_losses[0].item() * self.grad_acc:.4f}',
                'dsm': f'{step_losses[1].item() * self.grad_acc:.4f}',
                'fp': f'{step_losses[2].item() * self.grad_acc:.4f}'
            })
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        total_loss = sum(total_losses) / len(total_losses)
        dsm_loss = sum(dsm_losses) / len(dsm_losses)
        fp_loss = sum(fp_losses) / len(fp_losses)
        return total_loss, dsm_loss, fp_loss

    def train_step(self, x: torch.Tensor, use_fp = True):
        """train one step"""
        noise = torch.randn_like(x)
        x = x.clone()
        noise = noise.clone()
        t = self.sample_time(x.shape[0], self.eps_time)
        xt, true_score = self.fwd(x, noise, t)
        xt = xt.detach().requires_grad_(True)
        logp = self.mb_net(xt, t)
        pred_noise = self.noise_fn(logp, xt, t, self.fwd.vs)
        score = self.score_fn(logp, xt)
        mse_per_sample = F.mse_loss(pred_noise, noise, reduction='none')
        mse_per_sample = mse_per_sample.mean(dim=-1)
        lambda_vals = self.lambda_t(t)
        dsm_loss = (lambda_vals * mse_per_sample).mean()
        fp_loss = torch.tensor(0.0, device=self.device)
        if use_fp:
            with torch.no_grad():
                feats = self.fp_gate.fp_gate_features(xt.detach(), t, score.detach())
                gate_prob = self.fp_gate(feats).squeeze(-1)
            apply_fp_mask = gate_prob > 0.5
            fp_loss = torch.tensor(0.0, device=self.device)
            if apply_fp_mask.any():
                active_samps = torch.where(apply_fp_mask)[0]
                xt_active, t_active = self.subset_graph_data(xt, t, active_samps)
                seed1 = self.global_step * 1000
                r1 = self.fp_resid(self.mb_net, xt_active, t_active, self.fwd.vs, seed1)
                seed2 = self.global_step * 1001
                r2 = self.fp_resid(self.mb_net, xt_active, t_active, self.fwd.vs, seed2)
                fp_lambda_vals = self.lambda_t(t_active)
                fp_resid_product = self.fp_loss(r1, r2, alpha=self.fp_alpha)
                if fp_resid_product.dim() > 0:
                    fp_loss = (fp_lambda_vals * fp_resid_product).mean()
                else:
                    fp_loss = fp_resid_product
        total_loss = (dsm_loss + fp_loss) / self.grad_acc
        fp_loss = fp_loss / self.grad_acc
        dsm_loss = dsm_loss / self.grad_acc
        return total_loss, dsm_loss, fp_loss

    def gate_batch(self, pbar):
        self.fp_gate.train()
        epoch_losses = []
        for step, batch in enumerate(pbar):
            x = batch.to(self.device)
            step_loss = self.gate_step(x)
            step_loss.backward()
            clip_grad_norm_(self.fp_gate.parameters(), max_norm=1.0)
            self.gate_optim.step()
            self.gate_optim.zero_grad()
            epoch_losses.append(step_loss.item())
            pbar.set_postfix({'loss': f'{step_loss.item():.4f}'})
        self.fp_gate.eval()
        return sum(epoch_losses) / len(epoch_losses)

    def gate_step(self, x: torch.Tensor):
        noise = torch.randn_like(x)
        t = self.sample_time(x.shape[0], self.eps_time)
        xt, true_score = self.fwd(x, noise, t)
        xt = xt.detach().requires_grad_(True)
        logp = self.mb_net(xt, t)
        score = self.score_fn(logp, xt)
        with self.freeze_params(self.mb_net):
            r = self.fp_resid(self.mb_net, xt, t, self.fwd.vs)
        with torch.no_grad():
            feats = self.fp_gate.fp_gate_features(xt.detach(), t, score.detach())
        gate_prob = self.fp_gate(feats).squeeze(-1)
        with torch.no_grad():
            threshold = torch.quantile(r, self.fp_q)
            oracle_label = (r > threshold).float()
        gate_loss = self.gate_loss(gate_prob, oracle_label)
        return gate_loss

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

    @contextmanager
    def freeze_params(self, module):
        old_requires_grad = []
        for p in module.parameters():
            old_requires_grad.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(module.parameters(), old_requires_grad):
                p.requires_grad_(rg)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        checkpoint = {
            'current_epoch': epoch,
            'mb_net_state': self.mb_net.state_dict(),
            'fp_gate_state': self.fp_gate.state_dict(),
            'mb_optim_state': self.optim.state_dict(),
            'gate_optim_state': self.gate_optim.state_dict(),
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
            print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)