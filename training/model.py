import torch
import torch.nn as nn
from config import Config


class ElasticaEnergyNet(nn.Module):
    """Plain MLP energy network: (phi1, phi2, d) -> U.

    Loads derived from gradients:
      Fx     = -dU/dd
      M_left = -dU/dphi1 * (180/pi)
      M_right = +dU/dphi2 * (180/pi)
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

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
