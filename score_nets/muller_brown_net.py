import torch
import torch.nn as nn




class MullerBrownNet(nn.Module):
    """
    energy-based diffusion model that outputs log p(x, t)
    this is a conservative parameterization where:
        - Network outputs: log p(x, t) (scalar energy)
        - Score: s(x, t) = ∇_x log p(x, t)
        - Force: F(x) = -kBT * s(x, t=0)
    """
    def __init__(self, input_dim: int = 2, time_embed_dim: int = 64, hidden_dim: int = 128,
                 num_blocks: int = 3, activation: str ='silu'):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        self.time_embed = TimeEmbedding(time_embed_dim)
        #self.pos_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_proj = nn.Linear(input_dim + time_embed_dim, hidden_dim)
        #self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        compute log p(x, t)
        args:
            x: (batch_size, input_dim) positions
            t: (batch_size,) diffusion times in [0, 1]
        returns:
            log_p: (batch_size, 1) log probability
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() == 1 and t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        t_embed = self.time_embed(t)
        #t_proj = self.time_proj(t_embed)
        h = torch.cat([x, t_embed], dim=-1)
        h = self.input_proj(h)
        h = self.act(h)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        log_p = self.output_proj(h)
        return log_p

    def score(self, x, t, create_graph=True):
        """
        compute score function: s(x, t) = ∇_x log p(x, t)
        args:
            x: (batch_size, input_dim) positions
            t: (batch_size,) or scalar diffusion times
        returns:
            score: (batch_size, input_dim) score vectors
        """
        x_input = x.detach().requires_grad_(True)
        log_p = self.forward(x_input, t)
        score = torch.autograd.grad(
            outputs=log_p.sum(),
            inputs=x_input,
            create_graph=create_graph
        )[0]
        return score

    def energy(self, x, t):
        """
        compute energy: E(x, t) = -log p(x, t)
        args:
            x: (batch_size, input_dim) positions
            t: (batch_size,) or scalar diffusion times
        returns:
            energy: (batch_size, 1) energy values
        """
        with torch.no_grad():
            log_p = self.forward(x, t)
        return -log_p

    def force_at_t0(self, x: torch.Tensor, kbt: float = 23.0, t_eval: float = 1e-5):
        """
        compute forces at t = 0 for molecular simulation
        F(x) = -kBT * ∇_x log p(x, t=0)
        args:
            x: (batch_size, input_dim) positions
            kbt: Temperature in energy units
        returns:
            force: (batch_size, input_dim) force vectors
        """
        t_eval = torch.full((x.shape[0],), t_eval, device=x.device)
        #t_zero = torch.zeros(x.shape[0], device=x.device)
        score_t0 = self.score(x, t_eval)
        return -kbt * score_t0


class ResidualBlock(nn.Module):
    """residual block with layer norm"""
    def __init__(self, hidden_dim: int, activation: str = 'silu'):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.linear2(x)
        return x + residual


class TimeEmbedding(nn.Module):
    """time embedding"""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (number of atoms, ) or scalar
        returns: (number of atoms, embed_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)
        return self.mlp(t.unsqueeze(-1))