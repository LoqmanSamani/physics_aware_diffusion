import torch
import torch.nn as nn
from trainers.md_dist_trainer import MDDistillationTrainer
from data.loaders.md_sim_loader import TrajectoryParams, SyntheticMolecularDataset, create_dataloader
from score_nets.graph_energy_net import GraphEnergyNet
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from losses.distillation_loss import distillation_loss
from losses.fp_losses import fokker_planck_loss
from physics.fp_residuals import heavy_fp_residual, light_fp_residual
from physics.derive_score import score_from_energy
from physics.drift_score_gate import fp_gate



traj_params = TrajectoryParams(
    num_steps = 5,
    dt = 0.001,
    temp = 300.0,
    friction = 1.0,
    mass = 1.0,
    kb = 1.380649e-23,
    t_eval = 1e-5
)

t_model = GraphEnergyNet(
    atom_dim = 3,
    hidden_dim = 64,
    num_layers = 4,
    edge_dim = 50,
    num_heads = 2,
    dropout = 0.1
)

dataset = SyntheticMolecularDataset(
    n_molecules = 100,
    n_atom_types = 3,
    min_atoms = 5,
    max_atoms = 10,
    seed =  42,
    use_atom_types = True,
    include_trajectories = True,
    traj_params = traj_params,
    vary_traj_params = False,
    teacher_model = t_model
)
save_path = "./datasets"
filename = "md_dataset.pth"
dataset.save_to_disk(save_path, filename)
dataset_ = SyntheticMolecularDataset.load_from_disk(save_path, filename)

loader = create_dataloader(dataset_, batch_size = 32)
vs = LinearVS(beta_min = 0.1, beta_max= 20.0)

fwd = ForwardVP(vs)


s_model = GraphEnergyNet(
    atom_dim = 3,
    hidden_dim = 32,
    num_layers = 2,
    edge_dim = 50,
    num_heads = 1,
    dropout = 0.0
)



optim = torch.optim.AdamW(
    s_model.parameters(),
    lr = 1e-4,
    weight_decay = 1e-4,
    betas = (0.9, 0.999)
)

trainer = MDDistillationTrainer(
    teacher_enet = t_model,
    student_enet = s_model,
    forward_vp = fwd,
    data_loader = loader,
    optimizer = optim,
    fp_gate = fp_gate,
    fp_loss = fokker_planck_loss,
    dist_loss = distillation_loss,
    score_fn = score_from_energy,
    fp_residual = light_fp_residual,
    val_loader = None,
    epochs = 20,
    grad_acc = 1,
    checkpoint = 5,
    log_freq = 3,
    store_path = "./checkpoints",
    warmup_steps  = 20,
    rotation_augment = True,
    fp_alpha = 5e-4,
    mix_precision = True,
    gate_params = None,
    use_cached_trajectories = True,
    traj_loss_weight = 1.0,
    traj_freq = 5,
    eps_time = 1e-5
)

if __name__ == "__main__":
    print(sum(p.numel() for p in t_model.parameters()))
    print(sum(p.numel() for p in s_model.parameters()))
    losses = trainer()