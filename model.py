import torch
import torch.nn as nn
import math
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from typing import Callable
from torchvision.utils import save_image

class LinearVS(nn.Module):
    def __init__(self, num_steps: int = 1000, beta_start: float = 0.02,
                 beta_end: float = 2.0, start: float = 0.0, end: float = 1.0, *args):
        """the main scheduler used in this research"""
        super().__init__()
        self.num_steps = num_steps
        self.start = start
        self.end = end

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not (0.0 < beta_start < beta_end):
            raise ValueError(f"Must satisfy 0 < beta_start < beta_end")

        self.dt = (end - start) / (num_steps - 1)
        t = torch.linspace(start, end, num_steps)
        betas = beta_start + (beta_end - beta_start) * t / end
        integral_beta = beta_start * t + 0.5 * (beta_end - beta_start) * t ** 2 / end

        self.register_buffer("t", t)
        self.register_buffer("betas", betas)
        self.register_buffer("integral_beta", integral_beta)

    def get_variance(self, t_index: torch.Tensor) -> torch.Tensor:
        """get variance for vp sde: σ²(t) = 1 - exp(-∫₀ᵗ β(s) ds)"""
        return 1.0 - torch.exp(-self.integral_beta[t_index])

    def get_std(self, t_index: torch.Tensor) -> torch.Tensor:
        """get standard deviation: σ(t) = √(1 - exp(-∫₀ᵗ β(s) ds))"""
        return torch.sqrt(self.get_variance(t_index))

    def get_drift_coeff(self, t_index: torch.Tensor) -> torch.Tensor:
        """get drift coefficient: -0.5 * β(t)"""
        return -0.5 * self.betas[t_index]

    def get_diffusion_coeff(self, t_index: torch.Tensor) -> torch.Tensor:
        """get diffusion coefficient: √β(t)"""
        return torch.sqrt(self.betas[t_index])

class ForwardVP(nn.Module):
    """forward diffusion process of variance preserving sde"""
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, x0: torch.Tensor, noise: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:

        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps

        variance = self.vs.get_variance(t_index)
        signal_coeff = torch.sqrt(1.0 - variance)
        noise_coeff = torch.sqrt(variance)

        while signal_coeff.dim() < x0.dim():
            signal_coeff = signal_coeff.unsqueeze(-1)
            noise_coeff = noise_coeff.unsqueeze(-1)

        xt = signal_coeff * x0 + noise_coeff * noise

        return xt

class ReverseVP(nn.Module):
    """reverse diffusion process of variance preserving sde"""
    def __init__(self, variance_scheduler: nn.Module, *args) -> None:
        super().__init__()
        self.vs = variance_scheduler

    def forward(self, xt: torch.Tensor, noise_pred: torch.Tensor, t_index: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:

        assert t_index.min() >= 0 and t_index.max() < self.vs.num_steps

        dt = self.vs.dt
        beta_t = self.vs.betas[t_index]
        sigma_t = self.vs.get_std(t_index)

        while beta_t.dim() < xt.dim():
            beta_t = beta_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        score = -noise_pred / (sigma_t + 1e-8)
        drift = (-0.5 * beta_t * xt + beta_t * score) * dt
        diffusion = torch.sqrt(beta_t * dt) * noise
        xt_prev = (xt + drift + diffusion)

        return xt_prev

class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.downsample = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.mid1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid_attn = AttentionBlock(base_channels * 2)
        self.mid2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.up1 = ResBlock(base_channels * 4, base_channels, time_dim)
        self.up2 = ResBlock(base_channels, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.downsample(h2)
        h = self.mid1(h3, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        h = self.upsample(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, t_emb)
        h = self.up2(h, t_emb)
        h = self.act(self.out_norm(h))
        return self.out_conv(h)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class AttentionBlock(nn.Module):
    """self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, C, -1).transpose(1, 2)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).transpose(1, 2)
        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
        h = torch.bmm(attn, v)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(h)

class DiffusionTrainer(nn.Module):
    """trainer for the variance preserving diffusion model"""
    def __init__(self, score_net: torch.nn.Module, forward_vp: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                 loss_fn: Callable, epochs: int, grad_acc: int, checkpoint: int,
                 log_freq: int, store_path: str, device: str, warmup_steps: int = 0, *args) -> None:
        super().__init__()
        self.score_net = score_net
        self.forward_vp = forward_vp
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.device = device
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = (device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

    def _update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def forward(self) -> list:
        self.score_net.train()
        train_losses = []
        for epoch in range(self.epochs):
            train_losses_epoch = []
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                noise = torch.randn_like(x)
                time_ = torch.randint(0, self.forward_vp.vs.num_steps, (x.shape[0],), device=self.device)
                t_norm = time_.float() / (self.forward_vp.vs.num_steps - 1)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        noisy_x = self.forward_vp(x, noise, time_)
                    with torch.amp.autocast('cuda', enabled=False):
                        score = self.score_net(noisy_x, t_norm)
                    with torch.amp.autocast('cuda'):
                        loss = self.loss_fn(score, noise, time_, self.forward_vp.vs) / self.grad_acc
                    self.scaler.scale(loss).backward()
                else:
                    noisy_x = self.forward_vp(x, noise, time_)
                    score = self.score_net(noisy_x, t_norm)
                    loss = self.loss_fn(score, noise, time_, self.forward_vp.vs) / self.grad_acc
                    loss.backward()
                if (step + 1) % self.grad_acc == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    self._update_learning_rate()

                train_losses_epoch.append(loss.item() * self.grad_acc)
                pbar.set_postfix({'loss': loss.item() * self.grad_acc})
            if len(self.data_loader) % self.grad_acc != 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self._update_learning_rate()

            mean_train_loss = sum(train_losses_epoch) / len(train_losses_epoch)
            train_losses.append(mean_train_loss)

            if self.global_step >= self.warmup_steps:
                self.scheduler.step()
            # if (epoch + 1) % self.log_freq == 0:
            # lr = self.optimizer.param_groups[0]['lr']
            # print(f"\nEpoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | Train Loss: {mean_train_loss:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self._save_checkpoint(epoch + 1, mean_train_loss, train_losses)
            if mean_train_loss < self.best_loss:
                self.best_loss = mean_train_loss
                self._save_checkpoint(epoch + 1, mean_train_loss, train_losses, is_best=True)
        return train_losses

    def _save_checkpoint(self, epoch: int, loss: float, train_losses: list, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'score_net_state': self.score_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': train_losses,
            'scheduler': self.forward_vp.vs.state_dict(),
            'epochs': self.epochs
        }
        filename = "vp_best.pth" if is_best else f"vp_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch}")

class DiffusionSampler(nn.Module):
    """sampler for variance preserving sde diffusion"""
    def __init__(self, score_net: nn.Module, reverse_vp: nn.Module, output_size: Tuple[int, int],
                 batch_size: int, in_channels: int, device: str = "cuda", *args) -> None:
        super().__init__()
        self.device = device
        self.score_net = score_net.to(self.device)
        self.reverse_vp = reverse_vp.to(self.device)
        self.output_size = output_size
        self.batch_size = batch_size
        self.in_channels = in_channels

    def forward(self, store_path: str) -> torch.Tensor:
        noisy_x = torch.randn(
            self.batch_size, self.in_channels, self.output_size[0], self.output_size[1]
        ).to(self.device)
        self.score_net.eval()
        self.reverse_vp.eval()

        num_steps = self.reverse_vp.vs.num_steps
        iterator = reversed(range(num_steps))
        iterator = tqdm(list(iterator), desc="Sampling")

        with torch.no_grad():
            xt = noisy_x
            for t in iterator:
                time_ = torch.full((self.batch_size, ), t, dtype=torch.long).to(self.device)
                t_norm = time_.float() / (self.reverse_vp.vs.num_steps - 1)
                pred_noise = self.score_net(xt, t_norm)
                if t > 0:
                    noise = torch.randn_like(xt)
                else:
                    noise = torch.zeros_like(xt)
                xt = self.reverse_vp(xt, pred_noise, time_, noise)
            x0 = torch.clamp(xt, min=-1.0, max=1.0)
            os.makedirs(store_path, exist_ok=True)
            for i in range(x0.size(0)):
                img_path = os.path.join(store_path, f"img_{i + 1}.png")
                save_image((x0[i] + 1) / 2, img_path)
        return x0

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



def vp_noise_loss(pred_noise, true_noise, *args):
    return torch.mean((pred_noise - true_noise) ** 2)



vs = LinearVS(
    num_steps=1000,
    beta_start=0.1,
    beta_end=20.0,
    start=0.0,
    end=1.0,
).to("cuda")

forward_vp = ForwardVP(vs).to("cuda")

score_net = TinyUNet(
    in_channels=1,
    base_channels=64,
    time_dim=128,
).to("cuda")

optimizer = torch.optim.AdamW(
    score_net.parameters(),
    lr=0.002,
    weight_decay=0.0,
)



data_loader = get_mnist_subset_dataloader(
    batch_size=64,
    subset_fraction=1.0,
)


trainer = DiffusionTrainer(
    score_net=score_net,
    forward_vp=forward_vp,
    data_loader=data_loader,
    optimizer=optimizer,
    loss_fn=vp_noise_loss,
    epochs=100,
    grad_acc=1,
    checkpoint=20,
    log_freq=1,
    store_path="./model",
    device="cuda",
    warmup_steps=1000
)



train_losses = trainer()

reverse_vp = ReverseVP(vs).to("cuda")

score_net.eval()

sampler = DiffusionSampler(
    score_net=score_net,
    reverse_vp=reverse_vp,
    output_size=(28, 28),
    batch_size=9,
    in_channels=1
)

sampler("../mnist/results")



