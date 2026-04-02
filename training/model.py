import torch
import torch.nn as nn
from config import Config


class ResBlock(nn.Module):
    """Pre-activation residual block with LayerNorm."""
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

    Pure ResNet — no SIREN, no arc length, no theta.
    Arc length and theta are used ONLY in loss.py as physics teachers.

    Architecture:
        Linear embed  : 3   → hidden   (replaces SineLayer)
        ResBlock × N  : hidden → hidden
        Scalar head   : hidden → 4
    """

    def __init__(self,
                 hidden  = Config.HIDDEN_DIM,
                 n_blocks= Config.N_BLOCKS):
        super().__init__()

        # ── Input embedding ──────────────────────────────────────── #
        # Simple linear + GELU — appropriate for scalar tabular input
        self.embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
        )

        # ── Deep residual encoder ─────────────────────────────────── #
        self.encoder = nn.Sequential(
            *[ResBlock(hidden) for _ in range(n_blocks)]
        )

        # ── Scalar output head ────────────────────────────────────── #
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256,    128), nn.GELU(),
            nn.Linear(128,      4),            # Fx, Fy, ML, MR
        )

        # Kaiming initialisation — correct for GELU
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, phi):
        """
        phi : (B, 3)  normalised (phi1, phi2, d)
        Returns scalars : (B, 4)  Fx, Fy, ML, MR
        """
        z = self.encoder(self.embed(phi))      # (B, hidden)
        return self.head(z)                    # (B, 4)

    def count_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)