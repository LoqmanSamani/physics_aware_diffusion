import torch
import torch.nn as nn



class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.downsample = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.mid1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
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
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))

