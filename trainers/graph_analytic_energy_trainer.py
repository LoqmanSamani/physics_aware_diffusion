import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable


class AnalyticEnergyTrainer(nn.Module):
    """trainer for sanity-checking score net using analytic score targets"""
    def __init__(self, energy_net: nn.Module, score_from_energy: Callable, data_loader, optimizer,
                 score_fn: Callable, epochs: int, device: str):
        super().__init__()
        self.energy_net = energy_net.to(device)
        self.score_from_energy = score_from_energy
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.score_fn = score_fn
        self.epochs = epochs
        self.device = device

    def forward(self):
        self.energy_net.train()
        losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                x_t = batch.pos.to(self.device)
                atom = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                batch_vec = batch.batch.to(self.device)
                t = batch.t.to(self.device).squeeze(-1)
                sigma_t = batch.sigma_t.to(self.device).squeeze(-1)
                x_t = x_t.detach().requires_grad_(True)
                logp = self.energy_net(
                    coords =x_t,
                    atom_features = atom,
                    edge_index = edge_index,
                    t = t,
                    batch_idx = batch_vec
                )
                pred_score = self.score_from_energy(logp, x_t)

                sigma_per_node = sigma_t[batch_vec]
                true_score = self.score_fn(x_t, sigma_per_node)
                weights = (sigma_per_node.unsqueeze(-1) ** 2)
                weighted_diff = (pred_score - true_score) * weights
                loss = torch.mean(weighted_diff ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(mean_loss)
            print(f"Epoch {epoch + 1}: mean loss = {mean_loss:.6f}")
        return losses