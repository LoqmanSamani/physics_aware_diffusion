import torch.nn as nn


class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_dim=128):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, base_channels * 2)
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.down = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        temb = self.time_embed(t)
        temb = self.time_proj(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h_down = self.act(self.down(h))
        h_mid = self.act(self.mid(h_down + temb))
        h_up = self.act(self.up(h_mid))
        out = self.out(h_up + h)
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        return self.mlp(t.unsqueeze(-1).float())
