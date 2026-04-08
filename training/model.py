import torch
import torch.nn as nn
from config import Config


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ElasticaEnergyNet(nn.Module):
    def __init__(self, hidden=Config.HIDDEN_DIM, n_blocks=Config.N_BLOCKS):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )
        self.softplus = nn.Softplus(beta=1.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(self.embed(x))
        return self.softplus(self.head(z)).squeeze(-1)

    def energy_and_grad(self, x, create_graph=False):
        U = self.forward(x)
        g = torch.autograd.grad(
            outputs=U.sum(),
            inputs=x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return U, g

    def hessian(self, x):
        x = x.detach().requires_grad_(True)
        _, g = self.energy_and_grad(x, create_graph=True)
        B = x.shape[0]
        H = torch.zeros(B, 3, 3, device=x.device, dtype=x.dtype)
        for i in range(3):
            row = torch.autograd.grad(
                outputs=g[:, i].sum(),
                inputs=x,
                create_graph=False,
                retain_graph=(i < 2),
            )[0]
            H[:, i, :] = row
        return H

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
