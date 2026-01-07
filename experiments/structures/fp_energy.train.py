import torch
import torch.nn as nn
from losses.fp_loss import energy_fokker_planck_loss
from losses.dsm_losses import min_snr_weighted_loss
from score_nets.graph_energy_net import GraphEnergyNet
from trainers.fp_diffusion_trainer import FPEnergyTrainer
from physics.fokker_planck import weak_fp_residual, score_from_energy
from physics.fokker_planck_gate import FPGate
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from synthetic_molecular_dataset import SyntheticMolecularDataset, create_dataloader

dataset = SyntheticMolecularDataset(
    n_molecules = 1000,
    n_atom_types = 1,
    min_atoms = 10,
    max_atoms = 20,
    seed = 42
)
dataloader = create_dataloader(
    dataset = dataset,
    batch_size = 32,
    shuffle=True
)

val_loader = create_dataloader(
    dataset = dataset,
    batch_size = 8,
    shuffle=True
)

vs = LinearVS(
    num_steps=1000,
    beta_start=0.1,
    beta_end=20.0,
    start=0.0,
    end=1.0
)

fwd = ForwardVP(vs)
g_net = FPGate(hidden_dim=128)

e_net = GraphEnergyNet(
    atom_dim=1,
    hidden_dim=128,
    num_layers=3,
    dropout=0.2
)
#print(sum(p.numel() for p in score_net.parameters()))

e_optim = torch.optim.AdamW(
    e_net.parameters(),
    lr=1e-3,
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
    fp_loss = energy_fokker_planck_loss,
    dsm_loss = min_snr_weighted_loss,
    score_fn = score_from_energy,
    fp_residual = weak_fp_residual,
    epochs = 30,
    grad_acc = 2,
    checkpoint = 10,
    log_freq = 2,
    store_path = "./checkpoints",
    warmup_steps = 300,
    gate_epochs = 10,
    rotation_augment = False,
    gate_optimizer = g_optim,
    lambda_t = lambda t: 0.1
)

losses = trainer()

