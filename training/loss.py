import torch
import torch.nn.functional as F
import numpy as np
from config import Config



class ElasticaLoss:
    def __init__(self, dataset):
        self.y_mean = torch.from_numpy(dataset.y_mean)
        self.y_std = torch.from_numpy(dataset.y_std)
        self.x_mean = torch.from_numpy(dataset.x_mean)
        self.x_std = torch.from_numpy(dataset.x_std)
        self.t_mean = float(dataset.t_mean)
        self.t_std = float(dataset.t_std)
        self.arc_max = float(dataset.arc_max)
        self.scalar_weights = torch.tensor([Config.FX_WEIGHT, Config.FY_WEIGHT, Config.M_WEIGHT, Config.M_WEIGHT ])

    def _dn_theta(self, t):
        return t * self.t_std + self.t_mean

    def _dn_arc(self, arc):
        return arc * self.arc_max

    def _dn_x(self, x, device):
        return x * self.x_std.to(device) + self.x_mean.to(device)

    def _phys_to_norm(self, val_phys, out_idx, device):
        return (val_phys - self.y_mean[out_idx].to(device)) / self.y_std[out_idx].to(device)

    def _grad_norm_to_phys(self, g_comp, coord_idx, device):
        u_std = self.y_std[0].to(device)
        x_std = self.x_std[coord_idx].to(device)
        return g_comp * (u_std / x_std)

    @staticmethod
    def _deriv2(f, arc):
        h = (arc[:, 1:] - arc[:, :-1]).clamp(min=1e-8)
        df_ds = torch.zeros_like(f)

        h1 = h[:, :-1]
        h2 = h[:, 1:]
        df_ds[:, 1:-1] = (
            h1 ** 2 * f[:, 2:]
            - h2 ** 2 * f[:, :-2]
            - (h1 ** 2 - h2 ** 2) * f[:, 1:-1]
        ) / (h1 * h2 * (h1 + h2)).clamp(min=1e-8)

        h0 = h[:, 0]
        h1b = h[:, 1]
        df_ds[:, 0] = (
            -(2 * h0 + h1b) / (h0 * (h0 + h1b)).clamp(min=1e-8) * f[:, 0]
            + (h0 + h1b) / (h0 * h1b).clamp(min=1e-8) * f[:, 1]
            - h0 / (h1b * (h0 + h1b)).clamp(min=1e-8) * f[:, 2]
        )

        hN2 = h[:, -2]
        hN1 = h[:, -1]
        df_ds[:, -1] = (
            hN1 / (hN2 * (hN1 + hN2)).clamp(min=1e-8) * f[:, -3]
            - (hN1 + hN2) / (hN1 * hN2).clamp(min=1e-8) * f[:, -2]
            + (2 * hN1 + hN2) / (hN1 * (hN1 + hN2)).clamp(min=1e-8) * f[:, -1]
        )
        return df_ds

    @staticmethod
    def derive_from_theta(theta_phys, arc_phys, EI=Config.EI):
        dtheta_ds = ElasticaLoss._deriv2(theta_phys, arc_phys)
        M = EI * dtheta_ds
        ML = M[:, 0]
        MR = M[:, -1]

        dM_ds = ElasticaLoss._deriv2(M, arc_phys)
        theta_int = theta_phys[:, 1:-1]
        dM_int = dM_ds[:, 1:-1]

        A = torch.stack([
            torch.cos(theta_int),
            -torch.sin(theta_int)
        ], dim=-1)
        b = dM_int.unsqueeze(-1)
        sol = torch.linalg.lstsq(A, b).solution.squeeze(-1)

        Fy = sol[:, 0]
        Fx = sol[:, 1]
        return Fx, Fy, ML, MR, M, dtheta_ds

    @staticmethod
    def energy_from_theta(theta_phys, arc_phys, EI=Config.EI):
        dtheta_ds = ElasticaLoss._deriv2(theta_phys, arc_phys)
        density = 0.5 * EI * dtheta_ds ** 2
        ds_mid = 0.5 * (arc_phys[:, 2:] - arc_phys[:, :-2])
        return (density[:, 1:-1] * ds_mid).sum(dim=1)

    def __call__(self, model, x, y, arc_norm, theta_true, need_stiffness=False):
        device = x.device
        x_req = x.detach().requires_grad_(True)

        #U_pred_norm, g = model.energy_and_grad(x_req, create_graph=True)
        U_pred_norm = model(x_req)
        g = torch.autograd.grad(outputs=U_pred_norm.sum(), inputs=x_req,create_graph=True,
        retain_graph=True,
        )[0]
        x_phys = self._dn_x(x, device)
        d_phys = x_phys[:, 2].clamp(min=1e-8)

        energy_true = y[:, 0]
        fx_true = y[:, 1]
        fy_true = y[:, 2]
        m1_true = y[:, 3]
        m2_true = y[:, 4]

        theta_phys = self._dn_theta(theta_true)
        arc_phys = self._dn_arc(arc_norm)

        U_theta_phys = self.energy_from_theta(theta_phys, arc_phys)
        U_theta_norm = self._phys_to_norm(U_theta_phys, 0, device)

        loss_energy_label = F.mse_loss(U_pred_norm, energy_true)
        loss_energy_theta = F.mse_loss(U_pred_norm, U_theta_norm)

        ML_phys = Config.SIGN_M1 * self._grad_norm_to_phys(g[:, 0], 0, device) * ((180/np.pi))
        MR_phys = Config.SIGN_M2 * self._grad_norm_to_phys(g[:, 1], 1, device)  * ((180/np.pi))
        Fx_phys = Config.SIGN_FX * self._grad_norm_to_phys(g[:, 2], 2, device)
        Fy_phys = (MR_phys - ML_phys) / d_phys

        fx_pred = self._phys_to_norm(Fx_phys, 1, device)
        fy_pred = self._phys_to_norm(Fy_phys, 2, device)
        m1_pred = self._phys_to_norm(ML_phys, 3, device)
        m2_pred = self._phys_to_norm(MR_phys, 4, device)

        w = self.scalar_weights.to(device)
        loss_scalar = (
            w[0] * F.mse_loss(fx_pred, fx_true) +
            w[1] * F.mse_loss(fy_pred, fy_true) +
            w[2] * F.mse_loss(m1_pred, m1_true) +
            w[3] * F.mse_loss(m2_pred, m2_true)
        )

        loss_stiff = torch.tensor(0.0, device=device)
        if need_stiffness and Config.LAMBDA_STIFF > 0.0:
            H_rows = []
            for i in range(3):
                row = torch.autograd.grad(
                    outputs=g[:, i].sum(),
                    inputs=x_req,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if row is None:
                    row = torch.zeros_like(x_req)
                H_rows.append(row.unsqueeze(1))
            H = torch.cat(H_rows, dim=1)
            loss_stiff = (H ** 2).mean()

        total = (
            Config.W_ENERGY_LABEL * loss_energy_label +
            #Config.W_ENERGY_THETA * loss_energy_theta +
            Config.W_SCALAR * loss_scalar +
            Config.LAMBDA_STIFF * loss_stiff
        )

        return total, {
            "energy_label": loss_energy_label.item(),
            #"energy_theta": loss_energy_theta.item(),
            "scalar": loss_scalar.item(),
            "stiffness": loss_stiff.item(),
            "total": total.item(),
        }
