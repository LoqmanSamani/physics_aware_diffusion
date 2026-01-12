import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from typing import Callable, Tuple
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os



class TeacherTrainer(nn.Module):
    """teacher trainer with fokker-planck regularization applied through drift-score gating"""
    def __init__(
            self,
            energy_net: nn.Module,
            forward_vp: nn.Module,
            data_loader,
            optimizer: torch.optim.Optimizer,
            fp_gate: Callable,
            fp_loss: Callable, # fokker-planck loss
            dsm_loss: Callable, # denoising score matching loss
            noise_fn: Callable, # computes noise from energy
            score_fn: Callable,  # computes score from energy
            fp_residual: Callable, # computes weak fokker-planck residuals
            val_loader: None,
            epochs: int,
            device: torch.device | None = None,
            grad_acc: int = 1,
            checkpoint: int = 5,
            log_freq: int = 1,
            store_path: str = "./checkpoints",
            warmup_steps: int = 0,
            rotation_augment: bool = False, # if true molecules will be randomly augmented throw training
            fp_alpha: float = 5e-4,
            mix_precision: bool = True,
            lambda_t = lambda t: 1.0,  # time-dependent weighting λ(t)
            k: float = 1.0,
            t_max: float =  0.5,
            *args
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.energy_net = energy_net.to(self.device)
        self.forward_vp = forward_vp.to(self.device)
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.fp_gate = fp_gate
        self.fp_loss = fp_loss
        self.dsm_loss = dsm_loss
        self.score_fn = score_fn
        self.noise_fn = noise_fn
        self.fp_residual = fp_residual
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.rotation_augment = rotation_augment
        self.lambda_t = lambda_t
        self.fp_alpha = fp_alpha
        self.mix_precision = mix_precision
        self.k = k
        self.t_max = t_max
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = self.mix_precision and (self.device == torch.device("cuda"))
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        self.losses = {'total_losses': [], 'dsm_losses': [], 'fp_losses': [], 'val_losses': []}


    def forward(self):
        self.energy_net.train()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            mean_losses = self.train_batch(pbar, epoch)
            self.losses['total_losses'].append(mean_losses[0])
            self.losses['dsm_losses'].append(mean_losses[1])
            self.losses['fp_losses'].append(mean_losses[2])
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | "
                      f"Total Loss: {mean_losses[0]:.4f} | DSM Loss: {mean_losses[1]:.4f} "
                      f"| FP-Loss: {mean_losses[2]:.4f}")
                if self.val_loader is not None:
                    val_loss = self.validate()
                    self.losses['val_losses'].append(val_loss)
                    print(f" | Val Loss: {val_loss:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self.save_checkpoint(epoch + 1, mean_losses[0])
            if mean_losses[0] < self.best_loss:
                self.best_loss = mean_losses[0]
                self.save_checkpoint(epoch + 1, mean_losses[0], is_best=True)
        return self.losses

    def train_batch(self, pbar, epoch):
        """train one epoch of teacher model"""
        total_losses = []
        dsm_losses = []
        fp_losses = []
        for step, batch in enumerate(pbar):
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    step_losses = self.train_step(
                        x0, atom_features, edge_index, batch_idx, num_molecules
                    )
                self.scaler.scale(step_losses[0]).backward()
            else:
                step_losses = self.train_step(
                    x0, atom_features, edge_index, batch_idx, num_molecules
                )
                step_losses[0].backward()
            if (step + 1) % self.grad_acc == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.energy_net.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self.update_learning_rate()
            total_losses.append(step_losses[0].item() * self.grad_acc)
            dsm_losses.append(step_losses[1].item() * self.grad_acc)
            fp_losses.append(step_losses[2].item() * self.grad_acc)
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        total_loss = sum(total_losses) / len(total_losses)
        dsm_loss = sum(dsm_losses) / len(dsm_losses)
        fp_loss = sum(fp_losses) / len(fp_losses)
        return total_loss, dsm_loss, fp_loss

    def train_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch_idx: torch.Tensor, num_molecules: int):
        noise = torch.randn_like(x0)
        t_mol = self.sample_time(num_molecules) # (num_molecules,)
        t_atom = t_mol[batch_idx] # (num_atoms,)
        xt, true_score = self.forward_vp(x0, noise, t_atom)
        xt = xt.detach().requires_grad_(True)
        logp = self.energy_net(xt, atom_features, edge_index, t_atom, batch_idx)
        pred_noise = self.noise_fn(logp, xt) # derive score form energy and then convert it to noise
        pred_score = self.score_fn(logp, xt) # derive score form energy
        dsm_loss = self.lambda_t(t_atom.mean().item()) * self.dsm_loss(pred_noise, noise) # weighted dsm-loss
        fp_loss = torch.tensor(0.0, device=self.device)
        fp_mask = self.fp_gate(x0, t_atom, self.k, self.t_max)
        if fp_mask.any():
            xt_active, atom_feat_active, edge_idx_active, batch_active, t_active = self.subset_graph_data(
                    xt, atom_features, edge_index, t_atom, batch_idx, fp_mask, torch.unique(batch_idx[fp_mask])
            )
            seed1 = self.global_step * 1000
            r1 = self.fp_residual(
                self.energy_net, xt_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs, seed1
            )
            seed2 = self.global_step * 1001
            r2 = self.fp_residual(
                self.energy_net, xt_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs, seed2
            )
            var = self.forward_vp.vs.get_variance(t_active)
            fp_loss = self.lambda_t(t_active.mean().item()) * self.fp_loss(r1, r2, var) # weighted fp-loss
        total_loss = (dsm_loss + fp_loss) / self.grad_acc
        fp_loss = fp_loss / self.grad_acc
        dsm_loss = dsm_loss / self.grad_acc
        return total_loss, dsm_loss, fp_loss

    def validate(self):
        """validation without fp regularization"""
        #self.energy_net.eval()
        val_losses = []
        for batch in self.val_loader:
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            noise = torch.randn_like(x0)
            t_mol = self.sample_time(num_molecules)  # (num_molecules,)
            t_atom = t_mol[batch_idx]  # (num_atoms,)
            xt, true_score = self.forward_vp(x0, noise, t_atom)
            xt = xt.detach().requires_grad_(True)
            logp = self.energy_net(xt, atom_features, edge_index, t_atom, batch_idx)
            pred_noise = self.noise_fn(logp, xt)
            var = self.forward_vp.vs.get_variance(t_atom)
            loss = self.lambda_t(t_atom.mean().item()) * self.dsm_loss(pred_noise, noise) # weighted dsm-loss
            val_losses.append(loss.item())
        return sum(val_losses) / len(val_losses)

    def subset_graph_data(self, xt, atom_features, edge_index, t, batch_idx, active_atom_mask, active_mol_indices):
        xt_active = xt[active_atom_mask]
        atom_feat_active = atom_features[active_atom_mask]
        batch_active = batch_idx[active_atom_mask]
        old_to_new_mol = torch.full((batch_idx.max() + 1,), -1, dtype=torch.long, device=batch_idx.device)
        old_to_new_mol[active_mol_indices] = torch.arange(len(active_mol_indices), device=batch_idx.device)
        batch_active = old_to_new_mol[batch_active]
        edge_mask = active_atom_mask[edge_index[0]] & active_atom_mask[edge_index[1]]
        edge_idx_active = edge_index[:, edge_mask]
        old_to_new_atom = torch.full((xt.shape[0],), -1, dtype=torch.long, device=xt.device)
        old_to_new_atom[active_atom_mask] = torch.arange(active_atom_mask.sum(), device=xt.device)
        edge_idx_active = old_to_new_atom[edge_idx_active]
        t_active = t[active_atom_mask]
        return xt_active, atom_feat_active, edge_idx_active, batch_active, t_active

    def random_rotation_matrix(self, batch_size: int) -> torch.Tensor:
        """generate random 3D rotation matrices using QR decomposition"""
        M = torch.randn(batch_size, 3, 3, device=self.device)
        Q, R = torch.linalg.qr(M)
        det = torch.det(Q)
        Q = Q * det.view(-1, 1, 1).sign()
        return Q

    def update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        checkpoint = {
            'current_epoch': epoch,
            'energy_net_state': self.energy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'current_loss': loss,
            'losses': self.losses,
            'variance_scheduler': self.forward_vp.vs.state_dict(),
            'max_epochs': self.epochs,
            'base_larning_rate': self.base_lr,
            'global_step': self.global_step

        }
        filename = "fpd_best.pth" if is_best else f"fpd_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch} with loss {loss: .4f}")

    def sample_time(self, batch_size: int, eps: float = 1e-4) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)

