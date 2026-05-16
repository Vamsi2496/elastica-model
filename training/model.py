import math
import torch
import torch.nn as nn
from config import Config


class FourierEncoding(nn.Module):
    """Anisotropic random Fourier feature map: x → [sin(Bx), cos(Bx)].

    B is drawn with per-dimension sigmas (FOURIER_SIGMA_PHI for φ₁/φ₂,
    FOURIER_SIGMA_D for d) rather than a single isotropic sigma.  This
    allocates more frequency budget to d, where the snapping boundary creates
    sharp Fx variation over a narrow normalised range, while keeping lower
    frequencies for the smoother angular dependence.

    Output dimension: 2 * n_freqs  (sin and cos concatenated).
    """

    def __init__(self, input_dim: int, n_freqs: int,
                 sigma_phi: float, sigma_d: float):
        super().__init__()
        sigmas = torch.tensor([sigma_phi, sigma_phi, sigma_d])   # (3,)
        B = torch.randn(input_dim, n_freqs) * sigmas.unsqueeze(1)
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B                              # (N, n_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResBlock(nn.Module):
    """Two-layer residual block with GELU, no normalisation."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ElasticaEnergyNet(nn.Module):
    def __init__(self, hidden_layers=None):
        super().__init__()
        hidden_layers = hidden_layers or Config.HIDDEN_LAYERS

        # --- input stage ---
        if Config.FOURIER_FEATURES > 0:
            self.fourier = FourierEncoding(
                Config.INPUT_DIM, Config.FOURIER_FEATURES,
                sigma_phi=Config.FOURIER_SIGMA_PHI,
                sigma_d=Config.FOURIER_SIGMA_D,
            )
            in_dim = self.fourier.out_dim
        else:
            self.fourier = None
            in_dim = Config.INPUT_DIM

        # --- hidden layers ---
        blocks: list[nn.Module] = []
        prev_dim = in_dim
        for i, h in enumerate(hidden_layers):
            blocks.append(nn.Linear(prev_dim, h))
            blocks.append(nn.GELU())
            if Config.USE_RESIDUAL and prev_dim == h:
                # replace the plain Linear+GELU we just appended with a ResBlock
                blocks.pop()
                blocks.pop()
                blocks.append(ResBlock(h))
            prev_dim = h

        blocks.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*blocks)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # GELU ≈ linear in gain, so "linear" is the correct mode
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            x = self.fourier(x)
        return self.net(x).squeeze(-1)

    def energy_and_grad(self, x: torch.Tensor, create_graph: bool = False):
        x = x.requires_grad_(True)
        U = self.forward(x)
        g = torch.autograd.grad(U.sum(), x, create_graph=create_graph, retain_graph=True)[0]
        return U, g

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        _, g = self.energy_and_grad(x, create_graph=True)
        B = x.shape[0]
        H = torch.zeros(B, 3, 3, device=x.device, dtype=x.dtype)
        for i in range(3):
            row = torch.autograd.grad(g[:, i].sum(), x, create_graph=False, retain_graph=(i < 2))[0]
            H[:, i, :] = row
        return H

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
