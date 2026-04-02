import torch
import torch.nn.functional as F
from config import Config


class ElasticaLoss:
    """
    Total loss =
        weighted_MSE(scalars)        Fx weighted 3x
      + lambda_p * BC moment check   EI*dtheta_true/ds at s=0,L vs ML,MR pred
      + lambda_p * equilibrium ODE   dM/ds vs Fy*cos(theta) - Fx*sin(theta)
      + lambda_c * consistency       forces derived from theta_true vs scalar head

    All derivatives use 2nd-order accurate finite differences on non-uniform grids.
    theta_true is from HDF5 — physics teacher only, NOT a prediction target.
    """

    def __init__(self, dataset):
        self.y_mean  = torch.from_numpy(dataset.y_mean)
        self.y_std   = torch.from_numpy(dataset.y_std)
        self.t_mean  = float(dataset.t_mean)
        self.t_std   = float(dataset.t_std)
        self.arc_max = float(dataset.arc_max)

        # Scalar output loss weights: [Fx, Fy, ML, MR]
        self.scalar_weights = torch.tensor(
            [Config.FX_WEIGHT, 1.0, 1.0, 1.0]
        )

    # ------------------------------------------------------------------ #
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
        Compute df/ds at every node using 2nd-order finite differences
        on a non-uniform grid.

        f   : (B, N)  field values
        arc : (B, N)  arc length positions (physical units)

        Returns df_ds : (B, N)

        Interior nodes — non-uniform central difference:
            f'[i] = (h1²·f[i+1] - h2²·f[i-1] - (h1²-h2²)·f[i])
                    / (h1·h2·(h1+h2))
            where h1 = s[i]-s[i-1],  h2 = s[i+1]-s[i]

        Boundary nodes — 2nd-order one-sided:
            f'[0]  uses nodes 0,1,2  (forward)
            f'[-1] uses nodes -3,-2,-1  (backward)
        """
        h = (arc[:, 1:] - arc[:, :-1]).clamp(min=1e-8)    # (B, N-1)

        df_ds = torch.zeros_like(f)

        # ── Interior nodes (i = 1 … N-2) ───────────────────────────── #
        h1 = h[:, :-1]   # s[i]   - s[i-1]  (B, N-2)
        h2 = h[:, 1:]    # s[i+1] - s[i]    (B, N-2)

        df_ds[:, 1:-1] = (
            h1 ** 2 * f[:, 2:]
            - h2 ** 2 * f[:, :-2]
            - (h1 ** 2 - h2 ** 2) * f[:, 1:-1]
        ) / (h1 * h2 * (h1 + h2)).clamp(min=1e-8)

        # ── Left boundary (2nd-order one-sided forward) ─────────────── #
        # f'[0] = -(2h0+h1)/(h0*(h0+h1)) * f0
        #         + (h0+h1)/(h0*h1)       * f1
        #         - h0/(h1*(h0+h1))        * f2
        h0 = h[:, 0]     # s[1] - s[0]  (B,)
        h1b = h[:, 1]    # s[2] - s[1]  (B,)
        df_ds[:, 0] = (
            -(2 * h0 + h1b) / (h0 * (h0 + h1b)).clamp(1e-8) * f[:, 0]
            + (h0 + h1b)    / (h0 * h1b).clamp(1e-8)         * f[:, 1]
            - h0             / (h1b * (h0 + h1b)).clamp(1e-8) * f[:, 2]
        )

        # ── Right boundary (2nd-order one-sided backward) ───────────── #
        # f'[-1] = hN1/(hN2*(hN1+hN2))     * f[-3]
        #         - (hN1+hN2)/(hN1*hN2)     * f[-2]
        #         + (2*hN1+hN2)/(hN1*(hN1+hN2)) * f[-1]
        hN2 = h[:, -2]   # s[-2] - s[-3]  (B,)
        hN1 = h[:, -1]   # s[-1] - s[-2]  (B,)
        df_ds[:, -1] = (
            hN1               / (hN2 * (hN1 + hN2)).clamp(1e-8) * f[:, -3]
            - (hN1 + hN2)     / (hN1 * hN2).clamp(1e-8)         * f[:, -2]
            + (2 * hN1 + hN2) / (hN1 * (hN1 + hN2)).clamp(1e-8) * f[:, -1]
        )

        return df_ds                                        # (B, N)

    # ── Derive forces from TRUE theta ─────────────────────────────── #
    @staticmethod
    def derive_forces(theta_phys, arc_phys, EI=Config.EI):
        """
        Derive Fx, Fy, ML, MR purely from the TRUE theta field.
        Uses 2nd-order derivatives throughout.

        theta_phys : (B, N)  physical theta values (radians)
        arc_phys   : (B, N)  physical arc length positions
        Returns    : Fx, Fy, ML, MR  each (B,)
        """
        # dtheta/ds at every node — 2nd order
        dtheta_ds = ElasticaLoss._deriv2(theta_phys, arc_phys)    # (B, N)

        # M(s) = EI * dtheta/ds at every node
        M = EI * dtheta_ds                                         # (B, N)

        # Boundary moments: exact from 2nd-order boundary derivatives
        ML = M[:,  0]                                              # (B,)
        MR = M[:, -1]                                             # (B,)

        # dM/ds at every node — 2nd order
        dM_ds = ElasticaLoss._deriv2(M, arc_phys)                 # (B, N)

        # Equilibrium ODE at interior nodes:
        #   dM/ds = Fy*cos(theta) - Fx*sin(theta)
        #   [cos(theta) | -sin(theta)] * [Fy; Fx]^T = dM/ds
        theta_int = theta_phys[:, 1:-1]                           # (B, N-2)
        dM_int    = dM_ds[:, 1:-1]                                # (B, N-2)

        A   = torch.stack([
            torch.cos(theta_int),
            -torch.sin(theta_int)
        ], dim=-1)                                                 # (B, N-2, 2)
        b   = dM_int.unsqueeze(-1)                                # (B, N-2, 1)
        sol = torch.linalg.lstsq(A, b).solution.squeeze(-1)       # (B, 2)

        Fy = sol[:, 0]
        Fx = sol[:, 1]

        return Fx, Fy, ML, MR

    # ── Main loss call ─────────────────────────────────────────────── #
    def __call__(self, scalar_pred, scalar_true, theta_true, arc_norm):
        """
        scalar_pred : (B, 4)    model output         — normalised
        scalar_true : (B, 4)    HDF5 labels          — normalised
        theta_true  : (B, 201)  HDF5 theta field     — normalised (physics teacher)
        arc_norm    : (B, 201)  arc length            — normalised [0, 1]
        """
        device = scalar_pred.device

        # ── 1. Supervised scalar loss (Fx weighted 3x) ───────────────── #
        w           = self.scalar_weights.to(device)
        loss_scalar = (((scalar_pred - scalar_true) ** 2) * w).mean()

        # ── 2. Denormalise to physical units ──────────────────────────── #
        sp       = self._dn_scalar(scalar_pred, device)    # (B, 4)
        tp       = self._dn_theta(theta_true)              # (B, 201)
        arc_phys = self._dn_arc(arc_norm)                  # (B, 201)

        Fx = sp[:, 0]
        Fy = sp[:, 1]
        ML = sp[:, 2]
        MR = sp[:, 3]

        # ── 3. M(s) = EI * dtheta_true/ds — 2nd order ────────────────── #
        dtheta_ds = self._deriv2(tp, arc_phys)             # (B, 201)
        M_true    = Config.EI * dtheta_ds                  # (B, 201)

        # ── 4. BC moment check ────────────────────────────────────────── #
        # M(s=0) = ML_pred,  M(s=L) = MR_pred
        loss_ML = F.mse_loss(M_true[:,  0], ML)
        loss_MR = F.mse_loss(M_true[:, -1], MR)

        # ── 5. Equilibrium ODE — 2nd order dM/ds ─────────────────────── #
        # dM/ds = Fy_pred*cos(theta) - Fx_pred*sin(theta)
        dM_ds     = self._deriv2(M_true, arc_phys)         # (B, 201)
        theta_int = tp[:, 1:-1]                            # (B, 199)
        dM_int    = dM_ds[:, 1:-1]                         # (B, 199)
        rhs       = (Fy.unsqueeze(1) * torch.cos(theta_int)
                   - Fx.unsqueeze(1) * torch.sin(theta_int))  # (B, 199)
        loss_eq   = F.mse_loss(dM_int, rhs)

        # ── 6. Consistency ────────────────────────────────────────────── #
        Fx_d, Fy_d, ML_d, MR_d = self.derive_forces(tp, arc_phys)
        
        # Normalise each term by its std so all are dimensionless
        y_std = self.y_std.to(device)          # (4,) — [Fx_std, Fy_std, ML_std, MR_std]
        loss_cons = (F.mse_loss(Fx, Fx_d) / (y_std[0] ** 2 + 1e-8) +
                     F.mse_loss(Fy, Fy_d) / (y_std[1] ** 2 + 1e-8) +
                     F.mse_loss(ML, ML_d) / (y_std[2] ** 2 + 1e-8) +
                     F.mse_loss(MR, MR_d) / (y_std[3] ** 2 + 1e-8))

        # ── 7. Total ──────────────────────────────────────────────────── #
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