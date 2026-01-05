import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from torch_scatter import scatter_mean, scatter_std
from typing import Callable, Optional
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os


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
            rr_matrix: Callable,
            epochs: int,
            device: torch.device | None = None,
            grad_acc: int = 1,
            checkpoint: int = 5,
            log_freq: int = 1,
            store_path: str = "./checkpoints",
            warmup_steps: int = 0,
            gate_epochs: int = 10,
            rotation_augment: bool = False,
            gate_optimizer: Optional[torch.optim.Optimizer] = None,
            gate_loss: Optional[torch.nn.functional] = None,
            lambda_fp: int = 1.0,
            fp_threshold: float = 0.01,
            *args
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.energy_net = energy_net.to(self.device)
        self.fp_gate = fp_gate.to(self.device)
        self.forward_vp = forward_vp.to(self.device)
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.fp_loss = fp_loss
        self.dsm_loss = dsm_loss
        self.score_fn = score_fn
        self.fp_residual = fp_residual
        self.rr_matrix = rr_matrix
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.gate_epochs = gate_epochs
        self.rotation_augment = rotation_augment
        self.lambda_fp = lambda_fp
        self.fp_threshold = fp_threshold
        self.gate_loss = gate_loss or torch.nn.functional.binary_cross_entropy
        self.gate_optimizer = gate_optimizer or torch.oprim.Adam(self.fp_gate.parameters())
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = (self.device == torch.device("cuda"))
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        self.gate_losses = []
        self.losses = []


    def forward(self):
        self.score_net.train()
        self.fp_gate.train()
        for epoch in range(self.gate_epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.gate_epochs}")
            epoch_losses = self.gate_batch(pbar)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            self.gate_losses.append(mean_loss)

        self.fp_gate.eval()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            epoch_losses = self.train_batch(pbar)
            mean_loss = sum(epoch_losses) / len(epoch_losses)








    def train_batch(self, pbar):
        epoch_losses = []
        for step, batch in enumerate(pbar):
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    step_loss = self.train_step(x0, atom_features, edge_index, batch, num_molecules) / self.grad_acc
                self.scaler.scale(step_loss).backward()
            else:
                step_loss = self.train_step(x0, atom_features, edge_index, batch, num_molecules) / self.grad_acc
                step_loss.backward()
            if (step + 1) % self.grad_acc == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self._update_learning_rate()
            epoch_losses.append(step_loss.item() * self.grad_acc)
            pbar.set_postfix({'loss': f'{step_loss.item() * self.grad_acc:.4f}'})
            pbar.close()
        return epoch_losses

    def gate_batch(self, pbar):
        epoch_losses = []
        for step, batch in enumerate(pbar):
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            step_loss = self.gate_step(x0, atom_features, edge_index, batch, num_molecules)
            step_loss.backward()
            clip_grad_norm_(self.fp_gate.parameters(), max_norm=1.0)
            self.gate_optimizer.step()
            self.gate_optimizer.zero_grad()
            self.global_step += 1
            epoch_losses.append(step_loss.item())
            pbar.set_postfix({'loss': f'{step_loss.item():.4f}'})
            pbar.close()
        return epoch_losses


    def gate_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
               batch: torch.Tensor, num_molecules: int):
        noise = torch.randn_like(x0)
        t = torch.randint(1, self.forward_vp.vs.num_steps, (num_molecules,), device=self.device)
        t_per_atom = t[batch]
        t_norm_per_atom = t_per_atom.float() / (self.forward_vp.vs.num_steps - 1)
        xt, true_score = self.forward_vp(x0, noise, t_per_atom)
        logp = self.score_net(xt, atom_features, edge_index, t_norm_per_atom, batch)
        xt.requires_grad_(True)
        pred_score = self.score_fn(logp, xt)
        with torch.no_grad():
            gate_features = self.fp_gate.fp_gate_features_molecular(
                xt.detach(), t_norm_per_atom, pred_score.detach(), batch, num_molecules
            )
            gate_prob = self.fp_gate(gate_features)
            fp_residual1 = self.fp_residual(
                self.energy_net, x0, atom_features, edge_index, t, batch, self.forward_vp.vs
            )
            fp_residual2 = self.fp_residual(
                self.energy_net, x0, atom_features, edge_index, t, batch, self.forward_vp.vs
            )
            # fp_loss = self.fp_loss(fp_residual1, fp_residual2, x0.numel())
            residual_per_mol = scatter_mean((fp_residual1.abs() + fp_residual2.abs()) / 2, batch, dim=0)
            oracle_label = (residual_per_mol > self.fp_threshold).float().unsqueeze(-1)
        gate_loss = self.gate_loss(gate_prob, oracle_label)
        #self.gate_optimizer.zero_grad()
        #gate_loss.backward()
        #self.gate_optimizer.step()
        return gate_loss

    def train_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch: torch.Tensor, num_molecules: int) -> torch.Tensor:
        noise = torch.randn_like(x0)
        t = torch.randint(1, self.forward_vp.vs.num_steps, (num_molecules,), device=self.device)
        t_per_atom = t[batch]
        t_norm_per_atom = t_per_atom.float() / (self.forward_vp.vs.num_steps - 1)
        if self.rotation_augmentation:
            R = self.random_rotation_matrix(num_molecules, self.device)
            for mol_idx in range(num_molecules):
                mask = batch == mol_idx
                x0[mask] = x0[mask] @ R[mol_idx].T
                noise[mask] = noise[mask] @ R[mol_idx].T
        xt, true_score = self.forward_vp(x0, noise, t_per_atom)
        std_per_atom = self.forward_vp.vs.get_std(t_per_atom)
        variance = self.forward_vp.vs.get_variance(t_per_atom)
        while std_per_atom.dim() < noise.dim():
            std_per_atom = std_per_atom.unsqueeze(-1)
        logp = self.score_net(xt, atom_features, edge_index, t_norm_per_atom, batch)
        xt.requires_grad_(True)
        pred_score = self.score_fn(logp, xt)
        dsm_loss = self.dsm_loss(pred_score, true_score, variance, batch)
        fp_loss = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            gate_features = self.fp_gate.fp_gate_features(
                xt.detach(), t_norm_per_atom, pred_score.detach(), batch, num_molecules
            )
            gate_prob = self.fp_gate(gate_features)
        apply_fp_mask = (gate_prob > 0.5).squeeze(-1)
        if apply_fp_mask.any():
            active_mol_indices = torch.where(apply_fp_mask)[0]
            active_atom_mask = torch.isin(batch, active_mol_indices)
            x0_active, atom_feat_active, edge_idx_active, batch_active, t_active = self.subset_graph_data(
                x0, atom_features, edge_index, batch, active_atom_mask, active_mol_indices
            )
            fp_residual1 = self.fp_residual(
                self.energy_net, x0_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs
            )
            fp_residual2 = self.fp_residual(
                self.energy_net, x0_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs
            )
            fp_loss = self.fp_loss(fp_residual1, fp_residual2, x0_active.numel())
        return dsm_loss + self.lambda_fp * fp_loss

    def subset_graph_data(self, x0, atom_features, edge_index, batch, active_atom_mask, active_mol_indices):
        x0_active = x0[active_atom_mask]
        atom_feat_active = atom_features[active_atom_mask]
        batch_active = batch[active_atom_mask]
        old_to_new_mol = torch.full((batch.max() + 1,), -1, dtype=torch.long, device=batch.device)
        old_to_new_mol[active_mol_indices] = torch.arange(len(active_mol_indices), device=batch.device)
        batch_active = old_to_new_mol[batch_active]
        edge_mask = active_atom_mask[edge_index[0]] & active_atom_mask[edge_index[1]]
        edge_idx_active = edge_index[:, edge_mask]
        old_to_new_atom = torch.full((x0.shape[0],), -1, dtype=torch.long, device=x0.device)
        old_to_new_atom[active_atom_mask] = torch.arange(active_atom_mask.sum(), device=x0.device)
        edge_idx_active = old_to_new_atom[edge_idx_active]
        t_active = active_mol_indices
        return x0_active, atom_feat_active, edge_idx_active, batch_active, t_active

    def random_rotation_matrix(self, batch_size: int, device: str) -> torch.Tensor:
        """generate random 3D rotation matrices using QR decomposition"""
        M = torch.randn(batch_size, 3, 3, device=device)
        Q, R = torch.linalg.qr(M)
        det = torch.det(Q)
        Q = Q * det.view(-1, 1, 1).sign()
        return Q

    def _update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr













