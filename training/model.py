import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ── Building blocks ────────────────────────────────────────────────── #

class SineLayer(nn.Module):
    """SIREN layer with principled initialisation."""
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
    """Pre-activation residual block."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


# ── Main model ─────────────────────────────────────────────────────── #

class ElasticaINR(nn.Module):
    """
    Implicit Neural Representation for the Elastica BVP.

    Two parallel streams:
      Stream A — BC encoder   : (φ₁, φ₂, d)  → latent z  (B, H)
      Stream B — Arc encoder  : s             → pos_enc   (B, N, 2F)

    FiLM fusion:  z modulates pos_enc  →  θ(s)   per queried arc length

    Separate scalar head reads Fx, Fy, ML, MR directly from z
    (no arc length involved — forces are global properties).
    """

    def __init__(self,
                 hidden  = Config.HIDDEN_DIM,
                 n_blocks= Config.N_BLOCKS,
                 n_freq  = Config.N_FREQ,
                 omega   = Config.OMEGA):
        super().__init__()

        # ── Stream A: BC encoder ─────────────────────────────────────── #
        self.bc_embed   = SineLayer(3, hidden, omega=omega, is_first=True)
        self.bc_encoder = nn.Sequential(
            *[ResBlock(hidden) for _ in range(n_blocks)]
        )

        # ── Stream B: Arc length Fourier encoding ────────────────────── #
        # Fixed random frequencies — not trained, captures spatial detail
        freqs = torch.randn(1, n_freq) * 10.0          # (1, n_freq)
        self.register_buffer("freqs", freqs)
        pos_dim = 2 * n_freq                           # sin + cos

        # ── FiLM: z → scale γ and shift β for pos_enc ───────────────── #
        self.film_scale = nn.Linear(hidden, pos_dim)   # γ(z)
        self.film_shift = nn.Linear(hidden, pos_dim)   # β(z)

        # ── Theta decoder: fused features → θ(s) ───────────────────── #
        self.theta_decoder = nn.Sequential(
            nn.Linear(pos_dim, 256), nn.SiLU(),
            nn.Linear(256,     256), nn.SiLU(),
            nn.Linear(256,     128), nn.SiLU(),
            nn.Linear(128,       1),               # → scalar θ at each s
        )

        # ── Scalar head: z → Fx, Fy, ML, MR ────────────────────────── #
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256), nn.SiLU(),
            nn.Linear(256,    128), nn.SiLU(),
            nn.Linear(128,      4),
        )

    # ------------------------------------------------------------------ #
    def encode_bc(self, phi):
        """phi: (B, 3) → z: (B, H)"""
        return self.bc_encoder(self.bc_embed(phi))

    def encode_arc(self, s):
        """
        s   : (B, N)  arc length values, normalised to [0, 1]
        out : (B, N, 2*n_freq)
        """
        proj = s.unsqueeze(-1) * self.freqs.unsqueeze(0)   # (B, N, n_freq)
        return torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)        # (B, N, 2*n_freq)

    # ------------------------------------------------------------------ #
    def forward(self, phi, s):
        """
        phi : (B, 3)   normalised boundary conditions
        s   : (B, N)   arc length query positions ∈ [0, 1]

        Returns
        -------
        scalars : (B, 4)   Fx, Fy, ML, MR  (normalised)
        theta   : (B, N)   θ at each queried arc length (normalised)
        """
        # Stream A — BC latent
        z = self.encode_bc(phi)                            # (B, H)

        # Stream B — positional encoding of arc length
        pos = self.encode_arc(s)                           # (B, N, 2F)

        # FiLM: BC latent modulates arc-length features
        gamma = self.film_scale(z).unsqueeze(1)            # (B, 1, 2F)
        beta  = self.film_shift(z).unsqueeze(1)            # (B, 1, 2F)
        fused = gamma * pos + beta                         # (B, N, 2F)

        # Decode θ at each position
        theta   = self.theta_decoder(fused).squeeze(-1)    # (B, N)

        # Scalar outputs — from BC latent only
        scalars = self.scalar_head(z)                      # (B, 4)

        return scalars, theta

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)