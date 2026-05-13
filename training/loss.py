import numpy as np
import torch
import torch.nn.functional as F
from config import Config


class ElasticaLoss:
    """Loss for 2-input model (φ₁, φ₂) → U.

    Supervised outputs: Energy (direct), M_left and M_right (via ∂U/∂φ).
    Fx is not defined because d is not an input.
    """

    def __init__(self, dataset):
        self.y_mean = torch.from_numpy(dataset.y_mean)  # (3,): Energy, M_left, M_right
        self.y_std  = torch.from_numpy(dataset.y_std)
        self.x_mean = torch.from_numpy(dataset.x_mean)  # (2,): phi1, phi2
        self.x_std  = torch.from_numpy(dataset.x_std)

    def _grad_to_moment_phys(self, g_comp, coord_idx, device):
        """Convert normalised gradient component to physical moment (deg)."""
        return g_comp * (self.y_std[0].to(device) / self.x_std[coord_idx].to(device)) * (180.0 / np.pi)

    def _phys_to_norm(self, val_phys, out_idx, device):
        return (val_phys - self.y_mean[out_idx].to(device)) / self.y_std[out_idx].to(device)

    def __call__(self, model, x, y, need_stiffness=False):
        device = x.device
        x_req = x.detach().requires_grad_(True)
        U_pred_norm, g = model.energy_and_grad(x_req, create_graph=need_stiffness)

        energy_true = y[:, 0]
        m1_true     = y[:, 1]
        m2_true     = y[:, 2]

        loss_energy = F.mse_loss(U_pred_norm, energy_true)

        # g[:, 0] = ∂U_norm/∂φ₁_norm,  g[:, 1] = ∂U_norm/∂φ₂_norm
        ML_phys = Config.SIGN_M1 * self._grad_to_moment_phys(g[:, 0], 0, device)
        MR_phys = Config.SIGN_M2 * self._grad_to_moment_phys(g[:, 1], 1, device)

        m1_pred = self._phys_to_norm(ML_phys, 1, device)
        m2_pred = self._phys_to_norm(MR_phys, 2, device)

        mse_m1 = F.mse_loss(m1_pred, m1_true)
        mse_m2 = F.mse_loss(m2_pred, m2_true)
        loss_scalar = Config.M_WEIGHT * (mse_m1 + mse_m2)

        total = Config.W_ENERGY_LABEL * loss_energy + Config.W_SCALAR * loss_scalar
        return total, {
            "energy": float(loss_energy.item()),
            "M_left":  float(mse_m1.item()),
            "M_right": float(mse_m2.item()),
            "scalar":  float(loss_scalar.item()),
            "total":   float(total.item()),
        }
