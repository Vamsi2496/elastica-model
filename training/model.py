import torch
import torch.nn as nn
from config import Config


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, omega=30.0, is_first=False):
        super().__init__()
        self.omega  = omega
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_f, 1 / in_f)
            else:
                bound = (6 / in_f) ** 0.5 / omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ElasticaScalarNet(nn.Module):
    """
    Simple scalar surrogate:
        (φ₁, φ₂, d)  →  (Fx, Fy, ML, MR)

    Arc length and theta are NOT inputs to the model.
    They are used ONLY in the loss function as physics teachers.
    """

    def __init__(self,
                 hidden  = Config.HIDDEN_DIM,
                 n_blocks= Config.N_BLOCKS,
                 omega   = Config.OMEGA):
        super().__init__()

        # SIREN first layer — smooth encoding of BC space
        self.embed   = SineLayer(3, hidden, omega=omega, is_first=True)

        # ResNet encoder
        self.encoder = nn.Sequential(
            *[ResBlock(hidden) for _ in range(n_blocks)]
        )

        # Scalar output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256), nn.SiLU(),
            nn.Linear(256,    128), nn.SiLU(),
            nn.Linear(128,      4),            # Fx, Fy, ML, MR
        )

    def forward(self, phi):
        """
        phi : (B, 3)  normalised (φ₁, φ₂, d)
        Returns scalars : (B, 4)  Fx, Fy, ML, MR
        """
        z = self.encoder(self.embed(phi))     # (B, hidden)
        return self.head(z)                   # (B, 4)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)