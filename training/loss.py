import torch
import torch.nn.functional as F
from config import Config


class ElasticaLoss:
    """
    Total loss = w_scalar · MSE(scalars)
               + w_theta  · MSE(theta)
               + w_phys   · [moment-BC + equilibrium ODE]
               + w_cons   · [scalar_head vs physics-derived forces]

    Physics residuals use the actual non-uniform arc length spacing.
    All tensors are in normalised space; denormalised before physics eval.
    """

    def __init__(self, dataset):
        self.y_mean = torch.from_numpy(dataset.y_mean)   # (4,)
        self.y_std  = torch.from_numpy(dataset.y_std)    # (4,)
        self.t_mean = float(dataset.t_mean)
        self.t_std  = float(dataset.t_std)
        self.arc_max= float(dataset.arc_max)

    # ── Denormalisation helpers ────────────────────────────────────── #
    def _dn_scalar(self, y, device):
        return y * self.y_std.to(device) + self.y_mean.to(device)

    def _dn_theta(self, t):
        return t * self.t_std + self.t_mean

    def _dn_arc(self, arc):
        return arc * self.arc_max                        # → physical units

    # ── Physics-derived forces from θ(s) ──────────────────────────── #
    @staticmethod
    def derive_forces(theta_phys, arc_phys, EI=Config.EI):
        """
        theta_phys : (B, N)  physical θ
        arc_phys   : (B, N)  physical arc length
        Returns Fx, Fy, ML, MR derived from elastica ODEs.
        """
        ds = arc_phys[:, 1:] - arc_phys[:, :-1]         # (B, N-1)
        ds = ds.clamp(min=1e-8)

        # M(s) = EI · dθ/ds
        dtheta = (theta_phys[:, 1:] - theta_phys[:, :-1]) / ds
        M      = EI * dtheta                             # (B, N-1)

        # Boundary moments
        ML = M[:,  0]                                    # (B,)
        MR = M[:, -1]                                    # (B,)

        # dM/ds interior  (B, N-2)
        ds_mid  = (ds[:, 1:] + ds[:, :-1]) / 2
        dM_ds   = (M[:, 1:] - M[:, :-1]) / ds_mid       # (B, N-2)

        # Equilibrium: dM/ds = Fy·cosθ - Fx·sinθ
        # Interior theta values
        theta_int = theta_phys[:, 1:-1]                  # (B, N-2)
        cos_t     = torch.cos(theta_int)
        sin_t     = torch.sin(theta_int)

        # Batch least-squares per sample: [cos | -sin][Fy;Fx]^T = dM_ds
        A = torch.stack([ cos_t, -sin_t], dim=-1)        # (B, N-2, 2)
        b = dM_ds.unsqueeze(-1)                          # (B, N-2, 1)
        sol = torch.linalg.lstsq(A, b).solution          # (B, 2, 1)
        sol = sol.squeeze(-1)                            # (B, 2)

        Fy = sol[:, 0]
        Fx = sol[:, 1]

        return Fx, Fy, ML, MR

    # ── Main loss call ─────────────────────────────────────────────── #
    def __call__(self, scalar_pred, theta_pred,
                       scalar_true, theta_true, arc_norm):
        """
        scalar_pred : (B, 4)    normalised
        theta_pred  : (B, N)    normalised
        scalar_true : (B, 4)    normalised
        theta_true  : (B, N)    normalised
        arc_norm    : (B, N)    normalised arc length [0,1]
        """
        device = scalar_pred.device

        # ── 1. Supervised losses ────────────────────────────────────── #
        loss_scalar = F.mse_loss(scalar_pred, scalar_true)
        loss_theta  = F.mse_loss(theta_pred,  theta_true)

        # ── 2. Physics losses (denormalised) ────────────────────────── #
        sp       = self._dn_scalar(scalar_pred, device)   # (B, 4)
        tp       = self._dn_theta(theta_pred)             # (B, N)
        arc_phys = self._dn_arc(arc_norm)                 # (B, N)

        Fx_d = sp[:, 0];  Fy_d = sp[:, 1]
        ML_d = sp[:, 2];  MR_d = sp[:, 3]

        ds   = (arc_phys[:, 1:] - arc_phys[:, :-1]).clamp(1e-8)

        # M(s) = EI·dθ/ds
        dtheta  = (tp[:, 1:] - tp[:, :-1]) / ds
        M_curv  = Config.EI * dtheta                     # (B, N-1)

        # BC moment residuals
        loss_ML = F.mse_loss(M_curv[:,  0], ML_d)
        loss_MR = F.mse_loss(M_curv[:, -1], MR_d)

        # Equilibrium ODE residual: dM/ds vs Fy·cosθ - Fx·sinθ
        ds_mid   = (ds[:, 1:] + ds[:, :-1]) / 2
        dM_ds    = (M_curv[:, 1:] - M_curv[:, :-1]) / ds_mid  # (B, N-2)
        theta_int= tp[:, 1:-1]
        rhs      = (Fy_d.unsqueeze(1) * torch.cos(theta_int)
                  - Fx_d.unsqueeze(1) * torch.sin(theta_int))
        loss_eq  = F.mse_loss(dM_ds, rhs)

        # ── 3. Consistency loss ────────────────────────────────────── #
        # Derive forces from theta and compare to scalar head output
        Fx_drv, Fy_drv, ML_drv, MR_drv = self.derive_forces(tp, arc_phys)

        loss_cons = (F.mse_loss(Fx_d, Fx_drv) +
                     F.mse_loss(Fy_d, Fy_drv) +
                     F.mse_loss(ML_d, ML_drv) +
                     F.mse_loss(MR_d, MR_drv))

        # ── 4. Total ───────────────────────────────────────────────── #
        loss_phys = loss_ML + loss_MR + loss_eq

        total = (loss_scalar
                 + Config.LAMBDA_THETA * loss_theta
                 + Config.LAMBDA_PHYS  * loss_phys
                 + Config.LAMBDA_CONS  * loss_cons)

        breakdown = {
            "scalar"     : loss_scalar.item(),
            "theta"      : loss_theta.item(),
            "BC_moment"  : (loss_ML + loss_MR).item(),
            "equilibrium": loss_eq.item(),
            "consistency": loss_cons.item(),
            "total"      : total.item(),
        }
        return total, breakdown