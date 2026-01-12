import torch
from data.synthetic.molecular_dataset import SyntheticMolecularDataset, create_dataloader
from trainers.mol_score_trainer import MolScoreTrainer
from score_nets.graph_score_net import GraphScoreNet
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS
from losses.dsm_losses import min_snr_weighted_loss
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

score_net = GraphScoreNet(
    atom_dim=cfg['model']['atom_dim'],
    hidden_dim=cfg['model']['hidden_dim'],
    num_layers=cfg['model']['num_layers'],
    dropout=cfg['model']['dropout']
)
print(sum(p.numel() for p in score_net.parameters()))

optim = torch.optim.AdamW(
    score_net.parameters(),
    lr=cfg['training']['learning_rate'],
    weight_decay=cfg['training']['weight_decay'],
    betas=cfg['training']['betas']
)


trainer = MolScoreTrainer(
    score_net=score_net,
    forward_vp=fwd,
    data_loader=dataloader,
    optimizer=optim,
    loss_fn=min_snr_weighted_loss,
    epochs=cfg['training']['epochs'],
    device=cfg['experiment']['device'],
    grad_acc=cfg['training']['grad_accumulation'],
    checkpoint=cfg['logging']['checkpoint_freq'],
    log_freq=cfg['logging']['log_freq'],
    store_path=cfg['logging']['output_dir'],
    warmup_steps=cfg['training']['warmup_steps'],
    rotation_augmentation=cfg['training']['rotation_augmentation']
)

losses = trainer()