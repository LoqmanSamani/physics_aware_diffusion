import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models.score.tiny_unet import TinyUNet
from trainers.diffusion_trainer import DiffusionTrainer
from diffusion.forward import ForwardVP
from diffusion.schedules import LinearVS


batch_size = 32
epochs = 120
lr = 1e-4
grad_acc = 2
num_steps = 400
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mnist_subset_dataloader(batch_size: int = 32, subset_fraction: float = 0.1, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])

    full_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    subset_size = int(len(full_dataset) * subset_fraction)
    indices = torch.randperm(len(full_dataset))[:subset_size]
    subset = Subset(full_dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return loader

def vp_noise_loss(pred_noise, true_noise, t_index, scheduler):
    var = scheduler.get_variance(t_index)
    while var.dim() < pred_noise.dim():
        var = var.unsqueeze(-1)
    return torch.mean(var * (pred_noise - true_noise) ** 2)


vs = LinearVS(num_steps=400, beta_start=2e-4, beta_end=0.05, start=0.0, end=20.0).to(device)

forward_vp = ForwardVP(vs).to(device)
score_net = TinyUNet(in_channels=1, base_channels=32, time_dim=128).to(device)
optimizer = optim.AdamW(score_net.parameters(), lr=lr, weight_decay=1e-4)
data_loader = get_mnist_subset_dataloader(batch_size=32, subset_fraction=1.0)


trainer = DiffusionTrainer(
    score_net=score_net,
    forward_vp=forward_vp,
    data_loader=data_loader,
    optimizer=optimizer,
    loss_fn=vp_noise_loss,
    epochs=epochs,
    grad_acc=grad_acc,
    checkpoint=20,
    log_freq=1,
    store_path="./train_checkpoints",
    device=device
)

train_losses = trainer()







