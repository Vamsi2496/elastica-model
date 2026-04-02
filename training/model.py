import torch
import torch.nn as nn
from config import Config


class ResBlock(nn.Module):
    """Pre-activation residual block with LayerNorm + GELU."""
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


class ElasticaScalarNet(nn.Module):
    """
    Scalar surrogate:
        (phi1, phi2, d)  →  (Fx, Fy, ML, MR)

    No SIREN — scalar regression does not need sinusoidal embedding.
    No arc length / theta inputs — physics used only in loss.py.

    Architecture:
        Linear embed  : 3 → hidden  + GELU
        ResBlock × N  : hidden → hidden
        Scalar head   : hidden → 256 → 128 → 4
    """

    def __init__(self,
                 hidden   = Config.HIDDEN_DIM,
                 n_blocks = Config.N_BLOCKS):
        super().__init__()

        # ── Input embedding ───────────────────────────────────────────── #
        self.embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
        )

        # ── Residual encoder ──────────────────────────────────────────── #
        self.encoder = nn.Sequential(
            *[ResBlock(hidden) for _ in range(n_blocks)]
        )

        # ── Output head ───────────────────────────────────────────────── #
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256,    128), nn.GELU(),
            nn.Linear(128,      4),            # Fx, Fy, ML, MR
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, phi):
        """
        phi     : (B, 3)  normalised (phi1, phi2, d)
        returns : (B, 4)  normalised (Fx, Fy, ML, MR)
        """
        z = self.encoder(self.embed(phi))
        return self.head(z)

    def count_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)