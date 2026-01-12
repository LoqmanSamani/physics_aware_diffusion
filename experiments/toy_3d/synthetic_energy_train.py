import torch
from data.synthetic.molecular_dataset import SyntheticMolecularDataset, create_dataloader
from trainers.energy_trainer import MolEnergyTrainer
from score_nets.graph_energy_net import GraphEnergyNet
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from losses.dsm_losses import min_snr_weighted_loss
from physics.derive_noise import noise_from_energy
from pathlib import Path
from configs.load_config import load_config


project_root = Path(__file__).parent.parent.parent
config_path = project_root / "configs" / "synthetic_molecule_dataset.yaml"
cfg = load_config(str(config_path))


dataset = SyntheticMolecularDataset(
    n_molecules = cfg['dataset']['n_molecules'],
    n_atom_types = cfg['dataset']['n_atom_types'],
    min_atoms = cfg['dataset']['min_atoms'],
    max_atoms = cfg['dataset']['max_atoms'],
    seed = cfg['experiment']['seed']
)
dataloader = create_dataloader(
    dataset = dataset,
    batch_size = cfg['training']['batch_size'],
    shuffle=True
)

vs = LinearVS(
    beta_start=cfg['diffusion']['beta_start'],
    beta_end=cfg['diffusion']['beta_end']
)

fwd = ForwardVP(vs)

energy_net = GraphEnergyNet(
    atom_dim=cfg['model']['atom_dim'],
    hidden_dim=cfg['model']['hidden_dim'],
    num_layers=cfg['model']['num_layers'],
    dropout=cfg['model']['dropout'],
)
print(sum(p.numel() for p in energy_net.parameters()))

optim = torch.optim.AdamW(
    energy_net.parameters(),
    lr=cfg['training']['learning_rate'],
    weight_decay=cfg['training']['weight_decay'],
    betas=cfg['training']['betas']
)



trainer = MolEnergyTrainer(
    energy_net=energy_net,
    forward_vp=fwd,
    noise_fn = noise_from_energy,
    data_loader=dataloader,
    optimizer=optim,
    loss_fn=min_snr_weighted_loss,
    epochs=cfg['training']['epochs'],
    device=cfg['experiment']['device'],
    grad_acc=cfg['training']['grad_accumulation'],
    checkpoint=cfg['logging']['checkpoint_freq'],
    log_freq=cfg['logging']['log_freq'],
    store_path=cfg['logging']['output_dir'],
    warmup_steps=1000,
    rotation_augmentation=False
)

losses = trainer()