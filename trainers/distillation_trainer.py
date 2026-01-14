import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from typing import Callable, Tuple
from torch.nn.utils import clip_grad_norm_
from contextlib import contextmanager
from tqdm import tqdm
import os



class DistTrainer(nn.Module):
    """distillation trainer with fokker-planck regularization applied through gating"""
    def __init__(
            self,
            t_energy_net: nn.Module,
            energy_net: nn.Module,
            forward_vp: nn.Module,
            reverse_vp: nn.Module,
            dt, # student sampling step
            data_loader,
            optimizer: torch.optim.Optimizer,
            fp_gate: Callable,
            fp_loss: Callable, # fokker-planck loss
            dist_loss: Callable, # distillation loss
            score_fn: Callable, # computes score from energy
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
            lambda_t: Callable[[torch.Tensor], torch.Tensor] = lambda t: torch.exp(-t), # time-dependent weighting λ(t)
            k: float = 1.0,
            t_max: float =  0.5,
            *args
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_energy_net = t_energy_net.to(self.device)
        self.energy_net = energy_net.to(self.device)
        self.forward_vp = forward_vp.to(self.device)
        self.reverse_vp = reverse_vp.to(self.device)
        self.dt = dt
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.fp_gate = fp_gate
        self.fp_loss = fp_loss
        self.dist_loss = dist_loss
        self.score_fn = score_fn
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
        self.losses = {'total_losses': [], 'dist_losses': [], 'fp_losses': [], 'val_losses': []}
        self.t_min = self.forward_vp.eps


    def forward(self):
        self.t_energy_net.eval()
        self.energy_net.train()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            mean_losses = self.train_batch(pbar)
            self.losses['total_losses'].append(mean_losses[0])
            self.losses['dist_losses'].append(mean_losses[1])
            self.losses['fp_losses'].append(mean_losses[2])
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | "
                      f"Total Loss: {mean_losses[0]:.4f} | Dist Loss: {mean_losses[1]:.4f} "
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

    def train_batch(self, pbar):
        """train one epoch of student model"""
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
            pbar.set_postfix({
                'total': f'{step_losses[0].item() * self.grad_acc:.4f}',
                'dist': f'{step_losses[1].item() * self.grad_acc:.4f}',
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

    def train_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch_idx: torch.Tensor, num_molecules: int):
        """
        progressive distillation step for vp-sde:
            - teacher: two small reverse steps
            - student: one large reverse step
            - fp regularization applied only to student
        """
        # sample time and generate x_t using forward diffusion
        # ------------------------------------------------------------
        noise = torch.randn_like(x0)
        x0 = x0.clone()
        noise = noise.clone()
        # random rotations during training so that the network
        # learns rotational equivariance via data augmentation
        if self.rotation_augment:
            # autocast disabled for precise augmentation
            with torch.amp.autocast('cuda', enabled=False):
                R = self.random_rotation_matrix(num_molecules)
                for mol_idx in range(num_molecules):
                    mask = batch_idx == mol_idx
                    x0[mask] = x0[mask] @ R[mol_idx].T
                    noise[mask] = noise[mask] @ R[mol_idx].T
        t_mol = self.sample_time(num_molecules, self.t_min)  # (num_molecules,)
        t_atom = t_mol[batch_idx]  # (num_atoms,)
        lambda_val = self.lambda_t(t_atom.mean())
        xt, _ = self.forward_vp(x0, noise, t_atom)
        xt = xt.detach()  # x_t is fixed input now

        # two small reverse steps (markovian rollout) using teacher model
        # ------------------------------------------------------------
        # because we need gradients w.r.t. xt
        # we use freeze-prams function which freezes only teacher score net
        with self.freeze_params(self.t_energy_net):
            xt1 = xt.clone().requires_grad_(True)
            logp_t = self.t_energy_net(xt1, atom_features, edge_index, t_atom, batch_idx)
            t_score_t = self.score_fn(logp_t, xt1)
            # t -> 0 should be properly handled this way
            dt_small = torch.minimum(torch.full_like(t_atom, self.dt.item()), t_atom - self.t_min)
            noise1 = torch.randn_like(xt1)
            xt_dt = self.reverse_vp(xt1, t_score_t, t_atom, dt_small, noise=noise1, mode="sde")
            # t -> 0 should be properly handled this way
            t_atom_2 = t_atom - dt_small
            # allocate output
            xt_2dt = xt_dt.clone()
            valid_mask_2 = t_atom_2 > self.t_min
            xt_dt = xt_dt.detach().requires_grad_(True)
            logp_t2 = self.t_energy_net(xt_dt, atom_features, edge_index, t_atom_2, batch_idx)
            t_score_t2 = self.score_fn(logp_t2, xt_dt)
            if valid_mask_2.any():
                dt_small_2 = torch.minimum(torch.full_like(t_atom_2[valid_mask_2], self.dt.item()), t_atom_2[valid_mask_2] - self.t_min)
                noise2 = torch.randn_like(xt_dt[valid_mask_2])
                xt_2dt[valid_mask_2] = self.reverse_vp(
                    xt_dt[valid_mask_2], t_score_t2[valid_mask_2], t_atom_2[valid_mask_2], dt_small_2, noise=noise2, mode="sde"
                )

        # one large reverse step (2 * dt) using student model
        # ------------------------------------------------------------
        xt_st = xt.clone().requires_grad_(True)
        logp_st = self.energy_net(xt_st, atom_features, edge_index, t_atom, batch_idx)
        st_score = self.score_fn(logp_st, xt_st)
        valid_mask_st = t_atom > self.t_min
        xt_st_2dt = xt_st.clone()
        if valid_mask_st.any():
            dt_eff = torch.minimum(
                torch.full_like(t_atom[valid_mask_st], 2.0 * self.dt.item()), t_atom[valid_mask_st] - self.t_min
            )
            xt_st_2dt[valid_mask_st] = self.reverse_vp(
                xt_st[valid_mask_st], st_score[valid_mask_st], t_atom[valid_mask_st], dt_eff, mode="ode"
            )

        # distillation loss
        # ------------------------------------------------------------
        dist_loss = lambda_val * self.dist_loss(xt_st_2dt, xt_2dt.detach(), batch_idx)

        # fp-regularization only on student
        # ------------------------------------------------------------
        fp_loss = torch.tensor(0.0, device=self.device)
        fp_mask = self.fp_gate(x0, t_atom, self.forward_vp.vs, self.k, self.t_max)
        if fp_mask.any():
            xt_active, atom_feat_active, edge_idx_active, batch_active, t_active = self.subset_graph_data(
                xt_st, atom_features, edge_index, t_atom, batch_idx, fp_mask, torch.unique(batch_idx[fp_mask])
            )
            seed1 = self.global_step * 1000
            seed2 = self.global_step * 1001
            r1 = self.fp_residual(
                self.energy_net, xt_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs, seed1
            )
            r2 = self.fp_residual(
                self.energy_net, xt_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs, seed2
            )
            var = self.forward_vp.vs.get_variance(t_active)
            fp_lambda_val = self.lambda_t(t_active.mean())
            fp_loss = fp_lambda_val * self.fp_loss(r1, r2, batch_active, alpha=self.fp_alpha)  # weighted fp-loss
            # fp_loss = fp_lambda_val * self.fp_loss(r1, r2, batch_active, variance=var, alpha=self.fp_alpha)

        total_loss = (dist_loss + fp_loss) / self.grad_acc
        return total_loss, dist_loss / self.grad_acc, fp_loss / self.grad_acc

    def validate(self):
        """validation without fp regularization"""
        self.energy_net.eval()
        val_losses = []
        for batch in self.val_loader:
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            noise = torch.randn_like(x0)
            t_mol = self.sample_time(num_molecules, self.t_min)  # (num_molecules,)
            t_atom = t_mol[batch_idx]  # (num_atoms,)
            lambda_val = self.lambda_t(t_atom.mean())
            xt, true_score = self.forward_vp(x0, noise, t_atom)
            # freeze both teacher and student energy nets
            with self.freeze_params(self.t_energy_net):
                xt1 = xt.clone().requires_grad_(True)
                logp_t = self.t_energy_net(xt1, atom_features, edge_index, t_atom, batch_idx)
                t_score_t = self.score_fn(logp_t, xt1)
                dt_small = torch.minimum(torch.full_like(t_atom, self.dt.item()), t_atom - self.t_min)
                noise1 = torch.randn_like(xt1)
                xt_dt = self.reverse_vp(xt1, t_score_t, t_atom, dt_small, noise=noise1, mode="sde")
                t_atom_2 = t_atom - dt_small
                # allocate output
                xt_2dt = xt_dt.clone()
                valid_mask_2 = t_atom_2 > self.t_min
                xt_dt = xt_dt.detach().requires_grad_(True)
                logp_t2 = self.t_energy_net(xt_dt, atom_features, edge_index, t_atom_2, batch_idx)
                t_score_t2 = self.score_fn(logp_t2, xt_dt)
                if valid_mask_2.any():
                    dt_small_2 = torch.minimum(
                        torch.full_like(t_atom_2[valid_mask_2], self.dt.item()), t_atom_2[valid_mask_2] - self.t_min
                    )
                    noise2 = torch.randn_like(xt_dt[valid_mask_2])
                    xt_2dt[valid_mask_2] = self.reverse_vp(
                        xt_dt[valid_mask_2], t_score_t2[valid_mask_2], t_atom_2[valid_mask_2], dt_small_2, noise=noise2, mode="sde"
                    )
                with self.freeze_params(self.energy_net):
                    xt_st = xt.clone().requires_grad_(True)
                    logp_st = self.energy_net(xt_st, atom_features, edge_index, t_atom, batch_idx)
                    st_score = self.score_fn(logp_st, xt_st)
                    valid_mask_st = t_atom > self.t_min
                    xt_st_2dt = xt_st.clone()
                    if valid_mask_st.any():
                        dt_eff = torch.minimum(
                            torch.full_like(t_atom[valid_mask_st], 2.0 * self.dt.item()), t_atom[valid_mask_st] - self.t_min
                        )
                        xt_st_2dt[valid_mask_st] = self.reverse_vp(
                            xt_st[valid_mask_st], st_score[valid_mask_st], t_atom[valid_mask_st], dt_eff, mode="ode"
                        )
                    dist_loss = lambda_val * self.dist_loss(xt_st_2dt, xt_2dt.detach(), batch_idx)
                    val_losses.append(dist_loss.item())
        self.energy_net.train()
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
            'student_energy_net_state': self.energy_net.state_dict(),
            'teacher_energy_net_state': self.t_energy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'current_loss': loss,
            'losses': self.losses,
            'variance_scheduler': self.forward_vp.vs.state_dict(),
            'max_epochs': self.epochs,
            'base_larning_rate': self.base_lr,
            'global_step': self.global_step

        }
        filename = "dist_best.pth" if is_best else f"dist_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch} with loss {loss: .4f}")

    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)

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