import torch
from data.synthetic.molecular_dataset import SyntheticMolecularDataset, create_dataloader
from trainers.teacher_trainer import TeacherTrainer
from score_nets.energy_net import GraphEnergyNet
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from losses.dsm_losses import mse_loss, dsm_loss
from losses.fp_losses import snr_fokker_planck_loss, fokker_planck_loss
from physics.fp_residuals import heavy_fp_residual, light_fp_residual
from physics.derive_noise import noise_from_energy
from physics.drift_score_gate import fp_gate_
from pathlib import Path
from configs.load_config import load_config


project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "teacher_train.yaml"
cfg = load_config(str(config_path))


dataset = SyntheticMolecularDataset(
    n_molecules = 10000, #cfg['data_loader']['n_molecules'],
    n_atom_types = cfg['data_loader']['n_atom_types'],
    min_atoms = cfg['data_loader']['min_atoms'],
    max_atoms = cfg['data_loader']['max_atoms'],
    use_atom_types = cfg['data_loader']['use_atom_types'],
    seed = cfg['experiment']['seed']
)

val_dataset = SyntheticMolecularDataset(
    n_molecules = cfg['val_loader']['n_molecules'],
    n_atom_types = cfg['val_loader']['n_atom_types'],
    min_atoms = cfg['val_loader']['min_atoms'],
    max_atoms = cfg['val_loader']['max_atoms'],
    use_atom_types = cfg['val_loader']['use_atom_types'],
    seed = 21, #cfg['experiment']['seed']
)

dataloader = create_dataloader(
    dataset = dataset,
    batch_size = cfg['training']['batch_size'],
    shuffle=True
)


val_loader = create_dataloader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=True,
)

vs = LinearVS(
    beta_start=cfg['diffusion']['beta_start'],
    beta_end=cfg['diffusion']['beta_end']
)

fwd = ForwardVP(vs, 1e-5)

energy_net = GraphEnergyNet(
    atom_dim= cfg['model']['atom_dim'],
    hidden_dim=128, #cfg['model']['hidden_dim'],
    num_layers=4, #cfg['model']['num_layers'],
    edge_dim = 36,
    num_heads=2,
    dropout=cfg['model']['dropout'],
)
print(sum(p.numel() for p in energy_net.parameters()))

optim = torch.optim.AdamW(
    energy_net.parameters(),
    lr=1e-5, #cfg['training']['learning_rate'],
    weight_decay=cfg['training']['weight_decay'],
    betas=cfg['training']['betas']
)

trainer = TeacherTrainer(
    energy_net = energy_net,
    forward_vp = fwd,
    data_loader = dataloader,
    optimizer = optim,
    fp_gate = fp_gate_,
    fp_loss = snr_fokker_planck_loss,
    dsm_loss = mse_loss,
    noise_fn = noise_from_energy,
    fp_residual = heavy_fp_residual,
    val_loader = val_loader,
    epochs =  50,
    #device =
    grad_acc = 1,
    checkpoint = 10,
    log_freq = 3,
    store_path = "./checkpoints",
    warmup_steps = 400,
    rotation_augment = False,
    fp_alpha = 5e-4,
    mix_precision = True,
    k = 1.0,
    t_max = 0.4
)

losses = trainer()