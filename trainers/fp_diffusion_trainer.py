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
    """main trainer of the model. 
    used to train the original (teacher) model with fokker-planck regularization integrated"""
    def __init__(
            self,
            energy_net: nn.Module,
            fp_gate: nn.Module, # a simple mlp network used to specify where fp-regularization must be applied
            forward_vp: nn.Module,
            data_loader,
            optimizer: torch.optim.Optimizer,
            fp_loss: Callable, # fokker-planck loss
            dsm_loss: Callable, # denoising score matching loss
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
            gate_epochs: int = 10, # number of pretraining epochs in which fp-gate model will be trained
            rotation_augment: bool = False, # if true molecules will be randomly augmented throw training
            gate_optimizer: Optional[torch.optim.Optimizer] = None,
            gate_loss: Optional[torch.nn.functional] = None,
            lambda_t = lambda t: 1.0,  # time-dependent weighting λ(t)
            fp_quantile: float = 0.8, # top 30 percent get fokker-planck regularization
            fp_alpha: float = 5e-4,
            mix_precision: bool = True,
            *args
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.energy_net = energy_net.to(self.device)
        self.fp_gate = fp_gate.to(self.device)
        self.forward_vp = forward_vp.to(self.device)
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.fp_loss = fp_loss
        self.dsm_loss = dsm_loss
        self.score_fn = score_fn
        self.fp_residual = fp_residual
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.gate_epochs = gate_epochs
        self.rotation_augment = rotation_augment
        self.lambda_t = lambda_t
        self.fp_quantile = fp_quantile
        self.fp_alpha = fp_alpha
        self.gate_loss = gate_loss or nn.BCEWithLogitsLoss()
        self.gate_optimizer = gate_optimizer or torch.optim.Adam(self.fp_gate.parameters(), lr=1e-4)
        self.mix_precision = mix_precision
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = self.mix_precision and (self.device == torch.device("cuda"))
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        self.gate_losses = []
        self.losses = []
        self.val_losses = []
        self.gate_accuracy = []
        self.fp_application_rate = []


    def forward(self):
        self.energy_net.train()
        self.fp_gate.train()
        for epoch in range(self.gate_epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.gate_epochs}")
            epoch_losses = self.gate_batch(pbar)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            self.gate_losses.append(mean_loss)
            lr = self.gate_optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}/{self.gate_epochs} | LR: {lr:.2e} | Train Loss: {mean_loss:.4f}")

        self.fp_gate.eval()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            epoch_losses = self.train_batch(pbar)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(mean_loss)
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | Train Loss: {mean_loss:.4f}")
                if self.val_loader is not None:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    print(f" | Val Loss: {val_loss: .4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self.save_checkpoint(epoch + 1, mean_loss)
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.save_checkpoint(epoch + 1, mean_loss, is_best=True)
        return self.losses, self.gate_losses

    def train_batch(self, pbar):
        epoch_losses = []
        for step, batch in enumerate(pbar):
            #x0 = batch.coords.to(self.device)
            x0 = batch.pos.to(self.device)
            #atom_features = batch.atom_features.to(self.device)
            atom_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    step_loss = self.train_step(x0, atom_features, edge_index, batch_idx, num_molecules) / self.grad_acc
                self.scaler.scale(step_loss).backward()
            else:
                step_loss = self.train_step(x0, atom_features, edge_index, batch_idx, num_molecules) / self.grad_acc
                step_loss.backward()
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
            epoch_losses.append(step_loss.item() * self.grad_acc)
            pbar.set_postfix({'loss': f'{step_loss.item() * self.grad_acc:.4f}'})
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        return epoch_losses

    def gate_batch(self, pbar):
        epoch_losses = []
        for step, batch in enumerate(pbar):
            # x0 = batch.coords.to(self.device)
            x0 = batch.pos.to(self.device)
            #atom_features = batch.atom_features.to(self.device)
            atom_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            step_loss = self.gate_step(x0, atom_features, edge_index, batch_idx, num_molecules)
            step_loss.backward()
            clip_grad_norm_(self.fp_gate.parameters(), max_norm=1.0)
            self.gate_optimizer.step()
            self.gate_optimizer.zero_grad()
            self.global_step += 1
            epoch_losses.append(step_loss.item())
            pbar.set_postfix({'loss': f'{step_loss.item():.4f}'})
        return epoch_losses


    def gate_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
               batch: torch.Tensor, num_molecules: int):
        noise = torch.randn_like(x0)
        t = torch.randint(1, self.forward_vp.vs.num_steps, (num_molecules,), device=self.device)
        t_per_atom = t[batch]
        t_norm_per_atom = t_per_atom.float() / (self.forward_vp.vs.num_steps - 1)
        xt, true_score = self.forward_vp(x0, noise, t_per_atom)
        xt.requires_grad_(True)
        logp = self.energy_net(xt, atom_features, edge_index, t_norm_per_atom, batch)
        pred_noise = self.score_fn(logp, xt)
        fp_residual_ = self.fp_residual(
            self.energy_net, x0, atom_features, edge_index, t, batch, self.forward_vp.vs
        )
        fp_residual = (fp_residual_ - fp_residual_.mean(dim=0)) / (fp_residual_.std(dim=0) + 1e-6)
        #print("residuals", fp_residual)

        with torch.no_grad():
            features = self.fp_gate.fp_gate_features(
                xt.detach(), t, pred_noise.detach(), batch, num_molecules
            )
            norm_features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
        gate_logits = self.fp_gate(norm_features)
        with torch.no_grad():
            residual_per_mol = scatter_mean(fp_residual.abs(), batch, dim=0)
            #print('residual_per_mol')
            #print('-------------------------------------------------------')
            #print(residual_per_mol)
            #print('-------------------------------------------------------')
            fp_threshold = torch.quantile(residual_per_mol, self.fp_quantile)
            oracle_label = (residual_per_mol > fp_threshold).float().unsqueeze(-1)
            #print('this is oracle labels')
            #print('-------------------------------------------------------')
            #print(oracle_label)
            #print(oracle_label.unique(), oracle_label.float().mean())
            #print('-------------------------------------------------------')

        gate_loss = self.gate_loss(gate_logits, oracle_label)
        return gate_loss


    def train_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch: torch.Tensor, num_molecules: int) -> torch.Tensor:
        noise = torch.randn_like(x0)
        t = torch.randint(1, self.forward_vp.vs.num_steps, (num_molecules,), device=self.device)
        t_per_atom = t[batch]
        t_norm_per_atom = t_per_atom.float() / (self.forward_vp.vs.num_steps - 1)
        if self.rotation_augment:
            R = self.random_rotation_matrix(num_molecules)
            for mol_idx in range(num_molecules):
                mask = batch == mol_idx
                x0[mask] = x0[mask] @ R[mol_idx].T
                noise[mask] = noise[mask] @ R[mol_idx].T
        xt, true_score = self.forward_vp(x0, noise, t_per_atom)
        std_per_atom = self.forward_vp.vs.get_std(t_per_atom)
        variance = self.forward_vp.vs.get_variance(t_per_atom)
        while std_per_atom.dim() < noise.dim():
            std_per_atom = std_per_atom.unsqueeze(-1)
        xt.requires_grad_(True)
        logp = self.energy_net(xt, atom_features, edge_index, t_norm_per_atom, batch)
        pred_noise = self.score_fn(logp, xt)
        dsm_loss_ = self.dsm_loss(pred_noise, noise, variance, batch)
        #print(dsm_loss_)
        weight = self.lambda_t(t_norm_per_atom.mean().item())
        dsm_loss = weight * dsm_loss_ # weightened loss
        fp_loss = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            features = self.fp_gate.fp_gate_features(
                xt.detach(), t, pred_noise.detach(), batch, num_molecules
            )
            norm_features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
            gate_prob = self.fp_gate(norm_features)
        apply_fp_mask = (gate_prob > 0.5).squeeze(-1)
        if apply_fp_mask.any():
            active_mol_indices = torch.where(apply_fp_mask)[0]
            active_atom_mask = torch.isin(batch, active_mol_indices)
            x0_active, atom_feat_active, edge_idx_active, batch_active, t_active = self.subset_graph_data(
                x0, atom_features, edge_index, batch, active_atom_mask, active_mol_indices
            )
            # for efficiency reasons we compute fokker-planck residual only once
            fp_residual_ = self.fp_residual(
                self.energy_net, x0_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs
            )
            fp_residual = (fp_residual_ - fp_residual_.mean(dim=0)) / (fp_residual_.std(dim=0) + 1e-6)
            fp_loss_ = self.fp_loss(fp_residual, self.fp_alpha)
            t_norm_active = t_active.float() / (self.forward_vp.vs.num_steps - 1)
            weight_ = self.lambda_t(t_norm_active.mean().item())
            fp_loss = weight_ * fp_loss_  # weightened loss

            fp_residual_oracle_ = self.fp_residual(
                self.energy_net, x0_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs
            )
            fp_residual_oracle = (fp_residual_oracle_ - fp_residual_oracle_.mean(dim=0)) / (fp_residual_oracle_.std(dim=0) + 1e-6)

            #residual_per_mol = scatter_mean(fp_residual_oracle.abs(), batch, dim=0)
            residual_per_mol = scatter_mean(
                fp_residual_oracle.abs(), batch_active, dim=0, dim_size=len(active_mol_indices),
            )
            fp_threshold = torch.quantile(residual_per_mol, self.fp_quantile)
            #print(fp_threshold)
            oracle_label = (residual_per_mol > fp_threshold).float()
            #print(oracle_label.shape)
            gate_pred = (gate_prob.squeeze(-1) > 0.5).float()
            gate_pred_active = gate_pred[active_mol_indices]
            #print(gate_pred.shape)

            accuracy = (gate_pred_active == oracle_label).float().mean()
            self.gate_accuracy.append(accuracy.item())
            fp_rate = apply_fp_mask.float().mean().item()
            self.fp_application_rate.append(fp_rate)
        #print(dsm_loss, fp_loss)
        return dsm_loss + fp_loss

    def validate(self):
        """validation without fp regularization"""
        #self.energy_net.eval()
        val_losses = []
        for batch in self.val_loader:
            # x0 = batch.coords.to(self.device)
            x0 = batch.pos.to(self.device)
            # atom_features = batch.atom_features.to(self.device)
            atom_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            noise = torch.randn_like(x0)
            t = torch.randint(1, self.forward_vp.vs.num_steps, (num_molecules,), device=self.device)
            t_per_atom = t[batch_idx]
            t_norm_per_atom = t_per_atom.float() / (self.forward_vp.vs.num_steps - 1)
            xt, true_score = self.forward_vp(x0, noise, t_per_atom)
            variance = self.forward_vp.vs.get_variance(t_per_atom)
            xt.requires_grad_(True)
            logp = self.energy_net(xt, atom_features, edge_index, t_norm_per_atom, batch_idx)
            pred_noise = self.score_fn(logp, xt)
            loss_ = self.dsm_loss(pred_noise, noise, variance, batch_idx)
            weight = self.lambda_t(t_norm_per_atom.mean().item())
            loss = weight * loss_
            val_losses.append(loss.item())
        #self.energy_net.train()
        return sum(val_losses) / len(val_losses)

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
            'epoch': epoch,
            'energy_net_state': self.energy_net.state_dict(),
            'fp_gate_state': self.fp_gate.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'losses': self.losses,
            'gate_losses': self.gate_losses,
            'scheduler': self.forward_vp.vs.state_dict(),
            'epochs': self.epochs,
            'gate_accuracy': self.gate_accuracy,
            'fp_application_rate': self.fp_application_rate,
            'val_losses': self.val_losses
        }
        filename = "fpd_best.pth" if is_best else f"fpd_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch}")













