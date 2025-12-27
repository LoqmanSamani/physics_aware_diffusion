import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


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