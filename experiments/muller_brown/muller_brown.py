import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from diffusion.schedules import LinearVS
from diffusion.forward import ForwardVP
from diffusion.reverse import ReverseVP
from score_nets.muller_brown_net import MullerBrownNet
from data.loaders.muller_brown_data import (MullerBrownPotential, langevin_sampling, MolecularDataset,
                                            generate_muller_brown_data, MullerBrownEvaluator)
from trainers.muller_brown_trainer_ import MBTrainer
from samplers.muller_brown_sampler import MBSampler
from physics.derive_noise import noise_from_energy
from physics.derive_score import score_from_energy
from physics.drift_score_gate import fp_gate
from physics.fp_gate import MBFPGate
from physics.fp_residuals import mb_heavy_fp_residual, mb_light_fp_residual
from losses.dsm_losses import mse_loss
from losses.fp_losses import mb_fokker_planck_loss



device = 'cpu'
dataset, potential = generate_muller_brown_data(
    n_samples = 100,
    dt = 0.005,
    mass = 0.5,
    save_every = 10,
    kbt = 23.0,
    log_freq = 100,
    device = device,
    burn_in = 0,
    save_data = True,
    save_dir = './mb_data',
    filename = 'muller_brown_100k.pt',
    force_regenerate=False
)

dataloader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 0,
    drop_last = True
)

vs = LinearVS(beta_min = 0.1, beta_max = 20.0) # linear schedular with min and max beta (0.1, 20.0)

fwd = ForwardVP(vs) # forward diffusion which applies forward process of vp-sde (adding noise gradually to the data)

# we will train a relative small network
net = MullerBrownNet(
    input_dim = 2,
    time_embed_dim = 64,
    hidden_dim = 128,
    num_blocks = 3,
    activation ='silu'
)

print(sum(p.numel() for p in net.parameters())) # sum of all trainable parameters of the diffusion network (which in this case is a noise predictor)

optim = torch.optim.Adam([p for p in net.parameters()], lr=1e-4)

gate = MBFPGate(input_dim = 7, hidden_dim = 32)

trainer = MBTrainer(
    mb_net = net,
    fwd = fwd,
    data_loader = dataloader,
    optim = optim,
    fp_gate = gate,
    fp_loss = mb_fokker_planck_loss, # fokker planck loss
    noise_fn = noise_from_energy, # computes noise from energy
    score_fn = score_from_energy,
    fp_resid = mb_heavy_fp_residual, # computes weak fokker-planck residuals
    epochs = 30,
    device = torch.device(device),
    grad_acc = 2,
    checkpoint = 10,
    log_freq = 10,
    store_path  = "./mb1_train",
    warmup_steps = 3000,
    fp_alpha = 5e-4,
    eps_time = 1e-5,
    gate_epochs = 2,

)

losses = trainer()