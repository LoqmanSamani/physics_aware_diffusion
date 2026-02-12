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
from trainers.muller_brown_trainer import MBTrainer
from samplers.muller_brown_sampler import MBSampler
from physics.derive_noise import noise_from_energy
from physics.derive_score import score_from_energy
from physics.drift_score_gate import fp_gate
from physics.fp_gate import FPGate
from physics.fp_residuals import mb_heavy_fp_residual, mb_light_fp_residual
from losses.dsm_losses import mse_loss
from losses.fp_losses import mb_fokker_planck_loss



# example evaluation
def evaluate_model(model, dataset, potential, device='cpu'):
    """
    Complete evaluation pipeline

    Args:
        model: Trained diffusion model
        dataset: MolecularDataset with normalization stats
        potential: True potential for reference
    """
    evaluator = MullerBrownEvaluator(potential=potential, dataset_mean=dataset.mean, dataset_std=dataset.std, kbt=23.0)

    print("Generating reference samples from true potential...")
    reference_samples = langevin_sampling(
        potential=potential,
        n_steps=500000,
        save_every=5,
        device=device
    )

    print("\nGenerating IID samples from diffusion model...")
    iid_samples = evaluator.iid_sampling(sampler=model, n_samples=10000, device=device)

    print("\nGenerating simulation samples using learned score...")
    sim_samples = evaluator.simulation_sampling(model=model, n_steps=30000, n_parallel=100, device=device)

    print("\nVisualizing results...")
    evaluator.visualize_comparison(
        reference_samples=reference_samples,
        iid_samples=iid_samples,
        sim_samples=sim_samples,
        title_prefix=f"{model.__class__.__name__} - ",
        save_path=f"{model.__class__.__name__}_comparison.png"
    )

    return reference_samples, iid_samples, sim_samples

device = 'cuda'
dataset, potential = generate_muller_brown_data(
    n_samples = 1000,
    dt = 0.005,
    mass = 0.5,
    save_every = 100,
    kbt = 23.0,
    log_freq = 1000,
    device = device,
    burn_in = 1000,
    save_data = True,
    save_dir = './mb_data',
    filename = 'muller_brown_1k.pt',
    force_regenerate=False
)

dataloader = DataLoader(
    dataset,
    batch_size = 128,
    shuffle = True,
    num_workers = 0,
    drop_last = True
)



vs = LinearVS(beta_min = 0.1, beta_max = 20.0)

fwd = ForwardVP(vs)
rev = ReverseVP(vs)

net = MullerBrownNet(
    input_dim = 2,
    time_embed_dim = 64,
    hidden_dim = 128,
    num_blocks = 3,
    activation ='silu'
)

optim = torch.optim.Adam([p for p in net.parameters()], lr=1e-4)


trainer = MBTrainer(
    mb_net = net,
    fwd = fwd,
    data_loader = dataloader,
    optim = optim,
    fp_gate = fp_gate,
    fp_loss = mb_fokker_planck_loss,
    dsm_loss = mse_loss,
    noise_fn = noise_from_energy, # computes noise from energy
    fp_resid = mb_heavy_fp_residual, # computes weak fokker-planck residuals
    epochs = 10,
    device = torch.device(device),
    grad_acc = 1,
    checkpoint = 3,
    log_freq = 1,
    store_path  = "./mb1_train",
    warmup_steps = 10,
    fp_alpha = 5e-4,
    eps_time = 1e-5,
    gate_params = {"k": 1.2, "snr_min": 0.5, "t_scale": 0.0, "sharpness": 5.0, "eps": 0.1},
)
trainer()

sampler = MBSampler(
    model = net,
    rev = rev,
    eps_time = 1e-5,
    store_path = "./mb_samples",
    kbt = 23.0
)
samps = sampler.iid_sampler(n_samples = 1000, dt = 1e-3, device = 'cuda', store_trajectory = True, normalize_output = False)
lan_samps = sampler.langevin_simulator(n_parallel = 100, n_steps = 300, dt = 0.005, mass = 0.5,
                           gamma = 1.0, device ='cuda', save_every = 10, burn_in_ratio = 0.5)



