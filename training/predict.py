"""predict.py — interactive inference for the 2-input (φ₁, φ₂) energy model.

Outputs derived from the model:
  Energy  — direct network output U(φ₁, φ₂)
  M_left  = SIGN_M1 * ∂U/∂φ₁ * (y_std[E]/x_std[φ₁]) * (180/π)
  M_right = SIGN_M2 * ∂U/∂φ₂ * (y_std[E]/x_std[φ₂]) * (180/π)

Fx is not defined (d is not an input).
"""

import numpy as np
import torch

from config import Config
from model import ElasticaEnergyNet


class EnergyPredictor:
    def __init__(self, ckpt_path=Config.CKPT_BEST,
                 norm_stats_path=Config.NORM_STATS, device=None):
        self.device = device or Config.DEVICE
        st = np.load(norm_stats_path)
        self.x_mean = st["x_mean"].astype(np.float32)   # shape (2,): phi1, phi2
        self.x_std  = st["x_std"].astype(np.float32)
        self.y_mean = st["y_mean"].astype(np.float32)   # shape (3,): Energy, ML, MR
        self.y_std  = st["y_std"].astype(np.float32)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = ElasticaEnergyNet().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Loaded epoch {ckpt['epoch']} | "
              f"MLP [{', '.join(map(str, Config.HIDDEN_LAYERS))}] GELU | "
              f"params: {self.model.count_params():,}")

    def _norm_x(self, phi1, phi2):
        x = np.stack([
            np.asarray(phi1, np.float32).ravel(),
            np.asarray(phi2, np.float32).ravel(),
        ], axis=1)
        return torch.from_numpy((x - self.x_mean) / self.x_std).to(self.device)

    def _grad_to_phys_moment(self, g_comp, coord_idx):
        """Normalised gradient → physical moment (degrees)."""
        return g_comp * (self.y_std[0] / self.x_std[coord_idx]) * (180.0 / np.pi)

    def _hess_to_phys(self, H):
        """Convert 2×2 normalised Hessian to physical units."""
        out_std = self.y_std[0]
        in_std  = self.x_std                           # shape (2,)
        scale   = out_std / (in_std[:, None] * in_std[None, :])  # (2,2)
        return H * scale[None, :, :]

    def query(self, phi1, phi2, compute_stiffness=False):
        """Evaluate Energy, M_left, M_right (and optionally K) at (φ₁, φ₂).

        Args:
            phi1, phi2: boundary angles in degrees (scalar or array).
            compute_stiffness: if True, also return the 2×2 physical Hessian K.

        Returns:
            dict with keys: Energy, M_left, M_right [, K]
        """
        x = self._norm_x(phi1, phi2).detach().requires_grad_(True)
        U = self.model(x)
        g = torch.autograd.grad(U.sum(), x, create_graph=compute_stiffness)[0]

        U_phys = U.detach().cpu().numpy() * self.y_std[0] + self.y_mean[0]
        g_np   = g.detach().cpu().numpy()

        ML = float(Config.SIGN_M1 * self._grad_to_phys_moment(g_np[0, 0], 0))
        MR = float(Config.SIGN_M2 * self._grad_to_phys_moment(g_np[0, 1], 1))

        out = {"Energy": float(U_phys[0]), "M_left": ML, "M_right": MR}

        if compute_stiffness:
            H = np.zeros((1, 2, 2), dtype=np.float32)
            for i in range(2):
                row = torch.autograd.grad(
                    g[:, i].sum(), x,
                    create_graph=False,
                    retain_graph=(i < 1)
                )[0]
                H[:, i, :] = row.detach().cpu().numpy()
            out["K"] = self._hess_to_phys(H)[0]       # (2,2) physical Hessian

        return out

    def _loads_at(self, phi1, phi2):
        """Return [ML, MR] at a single point. Internal helper for FD."""
        r = self.query(phi1, phi2, compute_stiffness=False)
        return np.array([r["M_left"], r["M_right"]])

    def sensitivity(self, phi1, phi2):
        """2×2 Jacobian of [ML, MR] w.r.t. [φ₁ (deg), φ₂ (deg)] via Hessian.

        Note: uses second derivatives of U which were not directly supervised.
        Use sensitivity_fd() for more reliable estimates.

        Returns dict with keys: J (2×2), load_names, input_names.
        """
        res = self.query(phi1, phi2, compute_stiffness=True)
        K = res["K"]                                   # (2,2)
        dML = Config.SIGN_M1 * (180.0 / np.pi) * K[0, :]
        dMR = Config.SIGN_M2 * (180.0 / np.pi) * K[1, :]
        return {
            "J":           np.stack([dML, dMR]),
            "load_names":  ["ML", "MR"],
            "input_names": ["phi1 (deg)", "phi2 (deg)"],
        }

    def sensitivity_fd(self, phi1, phi2, eps_phi: float = 0.5):
        """2×2 Jacobian via central finite differences on model outputs.

        Args:
            eps_phi: perturbation in degrees. Default 0.5°.

        Returns dict with keys: J (2×2), load_names, input_names.
        """
        phi1 = float(phi1)
        phi2 = float(phi2)
        J = np.zeros((2, 2), dtype=np.float64)
        for col, (dp1, dp2) in enumerate([(eps_phi, 0.0), (0.0, eps_phi)]):
            f_plus  = self._loads_at(phi1 + dp1, phi2 + dp2)
            f_minus = self._loads_at(phi1 - dp1, phi2 - dp2)
            J[:, col] = (f_plus - f_minus) / (2.0 * eps_phi)
        return {
            "J":           J,
            "load_names":  ["ML", "MR"],
            "input_names": ["phi1 (deg)", "phi2 (deg)"],
        }


def _print_jacobian(label, sens):
    J, loads, inputs = sens["J"], sens["load_names"], sens["input_names"]
    print(f"  {label}:")
    print(f"    {'':8s} {inputs[0]:>14s} {inputs[1]:>14s}")
    for i, ln in enumerate(loads):
        print(f"    {ln:8s} {J[i,0]:14.4e} {J[i,1]:14.4e}")


if __name__ == "__main__":
    predictor = EnergyPredictor()

    for _ in range(6):
        print(f"\n{'='*55}")
        phi1 = float(input("Enter phi1 (deg): "))
        phi2 = float(input("Enter phi2 (deg): "))
        print(f"\nInput: phi1={phi1}°  phi2={phi2}°")

        res = predictor.query(phi1, phi2, compute_stiffness=False)
        print(f"  Energy  = {res['Energy']:.6f}")
        print(f"  M_left  = {res['M_left']:.6f}")
        print(f"  M_right = {res['M_right']:.6f}")


