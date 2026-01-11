import torch
import torch.nn as nn
from losses.fp_loss import fokker_planck_loss
from losses.dsm_losses import mse_loss
from score_nets.graph_energy_net import GraphEnergyNet
from trainers.gate_energy_trainer1 import FPEnergyTrainer
from physics.fp_residuals import heavy_fp_residual
from physics.derive_score import score_from_energy
from physics.fp_gate import FPGate
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from synthetic_molecular_dataset import SyntheticMolecularDataset, create_dataloader

dataset = SyntheticMolecularDataset(
    n_molecules = 10000,
    n_atom_types = 1,
    min_atoms = 10,
    max_atoms = 20,
    seed = 42
)
dataloader = create_dataloader(
    dataset = dataset,
    batch_size = 64,
    shuffle=True
)

val_loader = create_dataloader(
    dataset = dataset,
    batch_size = 8,
    shuffle=True
)

vs = LinearVS(beta_start=0.1, beta_end=20.0, min_variance=1e-5)
fwd = ForwardVP(vs)
g_net = FPGate(hidden_dim=256)
e_net = GraphEnergyNet(
    atom_dim=1,
    hidden_dim=256,
    num_layers=4,
    dropout=0.1
)
print(sum(p.numel() for p in e_net.parameters()))
print(sum(p.numel() for p in g_net.parameters()))

e_optim = torch.optim.AdamW(
    e_net.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
g_optim = torch.optim.AdamW(
    g_net.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)


trainer = FPEnergyTrainer(
    energy_net = e_net ,
    fp_gate = g_net,
    forward_vp = fwd,
    data_loader = dataloader,
    val_loader = val_loader,
    optimizer = e_optim,
    fp_loss = fokker_planck_loss,
    dsm_loss = mse_loss,
    score_fn = score_from_energy,
    fp_residual = heavy_fp_residual,
    epochs = 50,
    grad_acc = 2,
    checkpoint = 10,
    log_freq = 5,
    store_path = "./checkpoints2",
    warmup_steps = 500,
    gate_epochs = 10,
    rotation_augment = False,
    gate_optimizer = g_optim,
    mix_precision = False,
    compute_oracle = True,
    lambda_t = lambda t: 1.0
)

losses = trainer()

