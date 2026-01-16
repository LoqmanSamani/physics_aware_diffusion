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
        self.mse = nn.MSELoss()

    def forward(self):
        self.energy_net.train()
        losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                x_t = batch["x_t"].to(self.device)
                atom = batch["atom_features"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                t = batch["t"].to(self.device).squeeze(-1)
                sigma_t = batch["sigma_t"].to(self.device)
                B, N, _ = x_t.shape
                batch_vec = torch.arange(B, device=self.device).repeat_interleave(N)
                if edge_index.dim() == 3:
                    edge_list = []
                    for b in range(B):
                        edges_b = edge_index[b] + b * N
                        edge_list.append(edges_b)
                    edge_index_flat = torch.cat(edge_list, dim=1) if edge_list[0].size(1) > 0 else torch.empty(2, 0, dtype=torch.long, device=self.device)
                else:
                  edge_index_flat = edge_index
                x_t = x_t.detach().requires_grad_(True)
                logp = self.energy_net(
                    coords = x_t.view(B * N, 3),
                    atom_features = atom.view(B * N, -1),
                    edge_index = edge_index_flat,
                    t = t,
                    batch_idx = batch_vec
                )#.view(B, N, -1)
                pred_score = self.score_from_energy(logp, x_t)
                true_score = self.score_fn(x_t=x_t, sigma_t=sigma_t)
                if pred_score.size == true_score.size:
                    pred_score.view(B, N, -1)
                loss = self.mse(pred_score, true_score)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(mean_loss)
            print(f"Epoch {epoch + 1}: mean loss = {mean_loss:.6f}")
        return losses