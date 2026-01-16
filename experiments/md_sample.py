import torch
import torch.nn as nn
from score_nets.graph_energy_net import GraphEnergyNet
from diffusion.schedules import LinearVS
from diffusion.reverse import ReverseVP
from samplers.md_sampler import MDSampler
from physics.derive_score import score_from_energy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

atom_dim = 1
hidden_dim = 128
num_layers = 4

energy_net = GraphEnergyNet(
    atom_dim = atom_dim,
    hidden_dim = hidden_dim,
    num_layers = num_layers,
    edge_dim = 50,
    num_heads = 8,
    dropout = 0.2
)

vs = LinearVS(beta_min = 0.1, beta_max = 20.0)

reverse_vp = ReverseVP(vs).to(device)

sampler = MDSampler(
    energy_net=energy_net,
    reverse_vp=reverse_vp,
    score_fn=score_from_energy,
    store_path="./samples",
    device=device
)

print("Model parameters:", sum(p.numel() for p in energy_net.parameters()))

num_atoms = 5
atom_features = torch.randn(num_atoms, atom_dim, device=device)

# simple fully-connected graph (excluding self-loops)
edges = []
for i in range(num_atoms):
    for j in range(num_atoms):
        if i != j:
            edges.append([i, j])

edge_index = torch.tensor(edges, device=device).T  # (2, num_edges)
print("atom_features:", atom_features.shape)
print("edge_index:", edge_index.shape)


results = sampler.sample_iid(
    atom_features=atom_features,
    edge_index=edge_index,
    num_samples=2,
    dt=1e-3,
    sampling_mode="sde",
    store_trajectory=True
)

x0 = results["x0"]
trajectory = results["trajectory"]

print("Final samples:", x0.shape)           # (num_samples, num_atoms, 3)
print("Trajectory:", trajectory.shape)      # (num_steps, num_samples, num_atoms, 3)

assert not torch.isnan(x0).any()
assert torch.isfinite(x0).all()


x = x0[0]  # take first molecule
forces = sampler.get_forces(
    x=x,
    atom_features=atom_features,
    edge_index=edge_index,
    t_eval=1e-5
)

print("forces:", forces.shape)
print("force norm:", forces.norm(dim=1))


assert forces.shape == x.shape
assert torch.isfinite(forces).all()


md_results = sampler.simulate_langevin(
    x0=x,
    atom_features=atom_features,
    edge_index=edge_index,
    num_steps=1000,
    dt=1e-3,
    temp=300.0,
    save_frequency=50
)

traj = md_results["trajectory"]
vels = md_results["velocity_trajectory"]

print("MD trajectory:", traj.shape)
print("MD velocities:", vels.shape)



assert not torch.isnan(traj).any()