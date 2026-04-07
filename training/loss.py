import torch
import torch.nn.functional as F
from config import Config


class ElasticaLoss:
    """
    Train on Energy, and supervise force/moment gradients derived from energy.

    Targets in dataset y:
      y[:,0] = Energy
      y[:,1] = Fx
      y[:,2] = M_left
      y[:,3] = M_right
    """

    def __init__(self, dataset):
        self.y_mean = torch.from_numpy(dataset.y_mean)
        self.y_std = torch.from_numpy(dataset.y_std)
        self.x_std = torch.from_numpy(dataset.x_std)

    def _grad_norm_to_phys(self, g_comp, coord_idx, device):
        u_std = self.y_std[0].to(device)
        x_std = self.x_std[coord_idx].to(device)
        return g_comp * (u_std / x_std)

    def _phys_to_norm(self, val_phys, out_idx, device):
        return (val_phys - self.y_mean[out_idx].to(device)) / self.y_std[out_idx].to(device)

    def __call__(self, model, x, y, need_stiffness=False):
        device = x.device
        x_req = x.detach().requires_grad_(True)

        U_pred, g = model.energy_and_grad(x_req, create_graph=True)

        energy_true = y[:, 0]
        fx_true = y[:, 1]
        m1_true = y[:, 2]
        m2_true = y[:, 3]

        loss_energy = F.mse_loss(U_pred, energy_true)

        m1_phys = Config.SIGN_M1 * self._grad_norm_to_phys(g[:, 0], 0, device)
        m2_phys = Config.SIGN_M2 * self._grad_norm_to_phys(g[:, 1], 1, device)
        fx_phys = Config.SIGN_FX * self._grad_norm_to_phys(g[:, 2], 2, device)

        m1_pred = self._phys_to_norm(m1_phys, 2, device)
        m2_pred = self._phys_to_norm(m2_phys, 3, device)
        fx_pred = self._phys_to_norm(fx_phys, 1, device)

        loss_fx = F.mse_loss(fx_pred, fx_true)
        loss_m1 = F.mse_loss(m1_pred, m1_true)
        loss_m2 = F.mse_loss(m2_pred, m2_true)

        loss_stiff = torch.tensor(0.0, device=device)
        if need_stiffness and Config.LAMBDA_STIFF > 0.0:
            H_rows = []
            for i in range(3):
                row = torch.autograd.grad(
                    outputs=g[:, i].sum(),
                    inputs=x_req,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                H_rows.append(row.unsqueeze(1))
            H = torch.cat(H_rows, dim=1)
            loss_stiff = (H ** 2).mean()

        total = (
            Config.W_ENERGY * loss_energy +
            Config.W_FX * loss_fx +
            Config.W_M1 * loss_m1 +
            Config.W_M2 * loss_m2 +
            Config.LAMBDA_STIFF * loss_stiff
        )

        return total, {
            "energy": loss_energy.item(),
            "Fx": loss_fx.item(),
            "M_left": loss_m1.item(),
            "M_right": loss_m2.item(),
            "stiffness": loss_stiff.item(),
            "total": total.item(),
        }
