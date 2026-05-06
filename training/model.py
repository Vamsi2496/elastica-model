import torch
import torch.nn as nn
from config import Config


class ElasticaEnergyNet(nn.Module):
    """Plain MLP energy network: (φ₁, φ₂, d) → U.

    Hidden layers use GELU activation (smooth, good for autograd derivatives).
    Output layer is linear (energy is unconstrained).
    """

    def __init__(self, hidden_layers=None):
        super().__init__()
        hidden_layers = hidden_layers or Config.HIDDEN_LAYERS
        dims = [Config.INPUT_DIM] + list(hidden_layers) + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # GELU has near-unit gain, so "linear" is the correct Kaiming mode
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
