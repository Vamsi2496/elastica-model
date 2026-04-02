import torch
import torch.nn.functional as F
from config import Config


class ElasticaLoss:
    """
    Total loss =
        weighted_MSE(scalars)      Fx×3, Fy×0.1, ML×1, MR×1
      + lambda_p * BC moment       M at s=0,L  vs  ML,MR pred
      + lambda_p * equilibrium     dM/ds  vs  Fy·cos(θ) - Fx·sin(θ)
      + lambda_c * consistency     std-normalised Fx,Fy from θ_true

    theta_true — physics teacher only, NOT a prediction target.
    ML/MR excluded from consistency (covered by BC moment).
    All derivatives: 2nd-order accurate on non-uniform grid.
    """

    def __init__(self, dataset):
        self.y_mean  = torch.from_numpy(dataset.y_mean)
        self.y_std   = torch.from_numpy(dataset.y_std)
        self.t_mean  = float(dataset.t_mean)
        self.t_std   = float(dataset.t_std)
        self.arc_max = float(dataset.arc_max)

        self.scalar_weights = torch.tensor(
            [Config.FX_WEIGHT, Config.FY_WEIGHT, 1.0, 1.0]
        )

    # ── Denormalisation helpers ────────────────────────────────────── #
    def _dn_scalar(self, y, device):
        return y * self.y_std.to(device) + self.y_mean.to(device)

    def _dn_theta(self, t):
        return t * self.t_std + self.t_mean

    def _dn_arc(self, arc):
        return arc * self.arc_max

    # ── 2nd-order derivative on non-uniform grid ───────────────────── #
    @staticmethod
    def _deriv2(f, arc):
        """
        df/ds at every node — 2nd order accurate, non-uniform grid.
        f   : (B, N)
        arc : (B, N)
        returns df_ds : (B, N)
        """
        h     = (arc[:, 1:] - arc[:, :-1]).clamp(min=1e-8)
        df_ds = torch.zeros_like(f)

        # Interior — non-uniform central difference
        h1 = h[:, :-1]
        h2 = h[:, 1:]
        df_ds[:, 1:-1] = (
            h1 ** 2 * f[:, 2:]
            - h2 ** 2 * f[:, :-2]
            - (h1 ** 2 - h2 ** 2) * f[:, 1:-1]
        ) / (h1 * h2 * (h1 + h2)).clamp(min=1e-8)

        # Left boundary — 2nd order one-sided forward
        h0  = h[:, 0]
        h1b = h[:, 1]
        df_ds[:, 0] = (
            -(2 * h0 + h1b) / (h0  * (h0  + h1b)).clamp(1e-8) * f[:, 0]
            + (h0  + h1b)   / (h0  * h1b).clamp(1e-8)          * f[:, 1]
            - h0              / (h1b * (h0  + h1b)).clamp(1e-8) * f[:, 2]
        )

        # Right boundary — 2nd order one-sided backward
        hN2 = h[:, -2]
        hN1 = h[:, -1]
        df_ds[:, -1] = (
            hN1               / (hN2 * (hN1 + hN2)).clamp(1e-8) * f[:, -3]
            - (hN1 + hN2)     / (hN1 * hN2).clamp(1e-8)         * f[:, -2]
            + (2 * hN1 + hN2) / (hN1 * (hN1 + hN2)).clamp(1e-8) * f[:, -1]
        )

        return df_ds

    # ── Derive forces from TRUE theta ──────────────────────────────── #
    @staticmethod
    def derive_forces(theta_phys, arc_phys, EI=Config.EI):
        """
        Derives Fx, Fy, ML, MR purely from TRUE θ(s).
        Uses 2nd-order derivatives throughout.
        """
        dtheta_ds = ElasticaLoss._deriv2(theta_phys, arc_phys)
        M         = EI * dtheta_ds                                  # (B, N)

        ML = M[:,  0]
        MR = M[:, -1]

        dM_ds     = ElasticaLoss._deriv2(M, arc_phys)
        theta_int = theta_phys[:, 1:-1]
        dM_int    = dM_ds[:, 1:-1]

        A   = torch.stack([
            torch.cos(theta_int),
            -torch.sin(theta_int)
        ], dim=-1)                                                   # (B, N-2, 2)
        b   = dM_int.unsqueeze(-1)                                   # (B, N-2, 1)
        sol = torch.linalg.lstsq(A, b).solution.squeeze(-1)          # (B, 2)

        Fy = sol[:, 0]
        Fx = sol[:, 1]

        return Fx, Fy, ML, MR

    # ── Main loss ──────────────────────────────────────────────────── #
    def __call__(self, scalar_pred, scalar_true, theta_true, arc_norm):
        device = scalar_pred.device

        # 1. Supervised scalar loss
        w           = self.scalar_weights.to(device)
        loss_scalar = (((scalar_pred - scalar_true) ** 2) * w).mean()

        # 2. Denormalise
        sp       = self._dn_scalar(scalar_pred, device)
        tp       = self._dn_theta(theta_true)
        arc_phys = self._dn_arc(arc_norm)

        Fx = sp[:, 0];  Fy = sp[:, 1]
        ML = sp[:, 2];  MR = sp[:, 3]

        # 3. M(s) from TRUE theta
        dtheta_ds = self._deriv2(tp, arc_phys)
        M_true    = Config.EI * dtheta_ds

        # 4. BC moment check
        loss_ML = F.mse_loss(M_true[:,  0], ML)
        loss_MR = F.mse_loss(M_true[:, -1], MR)

        # 5. Equilibrium ODE
        dM_ds     = self._deriv2(M_true, arc_phys)
        theta_int = tp[:, 1:-1]
        dM_int    = dM_ds[:, 1:-1]
        rhs       = (Fy.unsqueeze(1) * torch.cos(theta_int)
                   - Fx.unsqueeze(1) * torch.sin(theta_int))
        loss_eq   = F.mse_loss(dM_int, rhs)

        # 6. Consistency — std-normalised, Fx and Fy only
        Fx_d, Fy_d, ML_d, MR_d = self.derive_forces(tp, arc_phys)
        y_std = self.y_std.to(device)
        loss_cons = (
            F.mse_loss(Fx, Fx_d) / (y_std[0] ** 2 + 1e-8) +
            F.mse_loss(Fy, Fy_d) / (y_std[1] ** 2 + 1e-8) + 
            F.mse_loss(ML, ML_d) / (y_std[2] ** 2 + 1e-8) +
            F.mse_loss(MR, MR_d) / (y_std[3] ** 2 + 1e-8)
        )

        # 7. Total
        loss_phys = loss_ML + loss_MR + loss_eq
        total = (loss_scalar
                 + Config.LAMBDA_PHYS * loss_phys
                 + Config.LAMBDA_CONS * loss_cons)

        return total, {
            "scalar"     : loss_scalar.item(),
            "BC_moment"  : (loss_ML + loss_MR).item(),
            "equilibrium": loss_eq.item(),
            "consistency": loss_cons.item(),
            "total"      : total.item(),
        }
