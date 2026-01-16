import torch
from pathlib import Path
from data.loaders.load_config import load_config
from losses.fp_losses import fokker_planck_loss
from losses.dsm_losses import mse_loss
from score_nets.graph_energy_net import GraphEnergyNet
from trainers.gate_energy_trainer1 import FPEnergyTrainer
from physics.fp_residuals import heavy_fp_residual, light_fp_residual
from physics.derive_score import score_from_energy
from physics.derive_noise import noise_from_energy
from physics.fp_gate import FPGate
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from data.loaders.molecular_dataset import (
    SyntheticMolecularDataset,
    create_dataloader,
)


cfg_path = Path("../configs/fp_energy_with_trainable_gate.yaml")
cfg = load_config(cfg_path)

torch.manual_seed(cfg["experiment"]["seed"])

dataset_cfg = cfg["dataset"]

dataset = SyntheticMolecularDataset(
    n_molecules=dataset_cfg["n_molecules"],
    n_atom_types=dataset_cfg["n_atom_types"],
    min_atoms=dataset_cfg["min_atoms"],
    max_atoms=dataset_cfg["max_atoms"],
    seed=dataset_cfg["seed"],
)

dataloader_cfg = cfg["dataloader"]
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=dataloader_cfg["batch_size"],
    shuffle=dataloader_cfg["shuffle"],
)

val_loader_cfg = cfg["val_dataloader"]
val_loader = create_dataloader(
    dataset=dataset,
    batch_size=val_loader_cfg["batch_size"],
    shuffle=val_loader_cfg["shuffle"],
)

vs_cfg = cfg["diffusion"]["variance_schedule"]
vs = LinearVS(
    beta_min=vs_cfg["beta_start"],
    beta_max=vs_cfg["beta_end"]
)
fwd = ForwardVP(vs)

energy_cfg = cfg["models"]["energy_net"]
e_net = GraphEnergyNet(
    atom_dim=energy_cfg["atom_dim"],
    hidden_dim=energy_cfg["hidden_dim"],
    num_layers=energy_cfg["num_layers"],
    edge_dim =energy_cfg["edge_dim"],
    num_heads =energy_cfg["num_heads"],
    dropout=energy_cfg["dropout"]
)

gate_cfg = cfg["models"]["fp_gate"]
g_net = FPGate(
    hidden_dim=gate_cfg["hidden_dim"]
)



eopt_cfg = cfg["optimizers"]["energy_optimizer"]
e_optim = torch.optim.AdamW(
    e_net.parameters(),
    lr=eopt_cfg["lr"],
    weight_decay=eopt_cfg["weight_decay"]
)

gopt_cfg = cfg["optimizers"]["gate_optimizer"]
g_optim = torch.optim.AdamW(
    g_net.parameters(),
    lr=gopt_cfg["lr"],
    weight_decay=gopt_cfg["weight_decay"]
)

trainer_cfg = cfg["trainer"]
logging_cfg = cfg["logging"]

trainer = FPEnergyTrainer(
    energy_net=e_net,
    fp_gate=g_net,
    forward_vp=fwd,
    data_loader=dataloader,
    val_loader=val_loader,
    optimizer=e_optim,
    fp_loss=fokker_planck_loss,
    dsm_loss=mse_loss,
    score_fn=score_from_energy,
    noise_fn=noise_from_energy,
    fp_residual=light_fp_residual, #heavy_fp_residual,
    epochs=trainer_cfg["epochs"],
    grad_acc=trainer_cfg["grad_acc"],
    checkpoint=logging_cfg["checkpoint_freq"],
    log_freq=logging_cfg["log_freq"],
    store_path=logging_cfg["store_path"],
    warmup_steps=trainer_cfg["warmup_steps"],
    gate_epochs=trainer_cfg["gate_epochs"],
    rotation_augment=trainer_cfg["rotation_augment"],
    gate_optimizer=g_optim,
    mix_precision=trainer_cfg["mixed_precision"],
    compute_oracle=trainer_cfg["compute_oracle"]
)

if __name__ == "__main__":
    print("Energy net params:", sum(p.numel() for p in e_net.parameters()))
    print("Gate net params:", sum(p.numel() for p in g_net.parameters()))
    losses = trainer()