import numpy as np
import torch

from config import Config
from model import ElasticaEnergyNet


class EnergyPredictor:
    def __init__(self, ckpt_path=Config.CKPT_BEST, norm_stats_path=Config.NORM_STATS, device=None):
        self.device = device or Config.DEVICE
        st = np.load(norm_stats_path)
        self.x_mean = st["x_mean"].astype(np.float32)
        self.x_std = st["x_std"].astype(np.float32)
        self.y_mean = st["y_mean"].astype(np.float32)
        self.y_std = st["y_std"].astype(np.float32)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = ElasticaEnergyNet().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        fourier_tag = f" Fourier×{Config.FOURIER_FEATURES}(σ_φ={Config.FOURIER_SIGMA_PHI},σ_d={Config.FOURIER_SIGMA_D})" if Config.FOURIER_FEATURES > 0 else ""
        print(f"Loaded epoch {ckpt['epoch']} | arch: [{', '.join(map(str, Config.HIDDEN_LAYERS))}]{fourier_tag} | params: {self.model.count_params():,}")

    def _norm_x(self, phi1, phi2, d):
        x = np.stack([np.asarray(phi1, np.float32).ravel(), np.asarray(phi2, np.float32).ravel(), np.asarray(d, np.float32).ravel()], axis=1)
        return torch.from_numpy((x - self.x_mean) / self.x_std).to(self.device)

    def _grad_to_phys(self, g):
        scale = self.y_std[0] / self.x_std
        return g * scale[None, :]

    def _hess_to_phys(self, H):
        out_std = self.y_std[0]
        in_std = self.x_std
        return H * (out_std / (in_std[:, None] * in_std[None, :]))[None, :, :]

    def query(self, phi1, phi2, d, compute_stiffness=True):
        x = self._norm_x(phi1, phi2, d).detach().requires_grad_(True)
        U = self.model(x)
        g = torch.autograd.grad(U.sum(), x, create_graph=compute_stiffness)[0]
        U_phys = U.detach().cpu().numpy() * self.y_std[0] + self.y_mean[0]
        g_phys = self._grad_to_phys(g.detach().cpu().numpy())
        d_phys = max(float(np.asarray(d).ravel()[0]), 1e-8)
        ML = float(Config.SIGN_M1 * g_phys[0, 0] * (180 / np.pi))
        MR = float(Config.SIGN_M2 * g_phys[0, 1] * (180 / np.pi))
        Fx = float(Config.SIGN_FX * g_phys[0, 2])
        Fy = float((MR - ML) / d_phys)
        out = {"Energy": float(U_phys[0]), "Fx": Fx, "Fy": Fy, "M_left": ML, "M_right": MR}
        if compute_stiffness:
            H = np.zeros((1, 3, 3), dtype=np.float32)
            for i in range(3):
                row = torch.autograd.grad(g[:, i].sum(), x, retain_graph=(i < 2))[0]
                H[:, i, :] = row.detach().cpu().numpy()
            out["K"] = self._hess_to_phys(H)[0]
        return out

    def _loads_at(self, phi1, phi2, d):
        """Return [ML, MR, Fx, Fy] at a single point. Internal helper."""
        r = self.query(phi1, phi2, d, compute_stiffness=False)
        return np.array([r["M_left"], r["M_right"], r["Fx"], r["Fy"]])

    def sensitivity(self, phi1, phi2, d):
        """4×3 Jacobian of [ML, MR, Fx, Fy] w.r.t. [φ₁ (deg), φ₂ (deg), d].

        WARNING: derived from the autograd Hessian of U (second derivatives of the
        network).  Because second derivatives were NOT supervised during training,
        this is qualitatively indicative but may be quantitatively unreliable,
        especially near the snapping boundary.  Use sensitivity_fd() for more
        reliable estimates.

        Returns a dict with:
          'J'           — (4, 3) ndarray, rows = [ML,MR,Fx,Fy], cols = [φ₁,φ₂,d]
          'load_names'  — ["ML", "MR", "Fx", "Fy"]
          'input_names' — ["phi1 (deg)", "phi2 (deg)", "d"]
        """
        res = self.query(phi1, phi2, d, compute_stiffness=True)
        K      = res["K"]                                       # (3,3) physical Hessian of U
        Fy     = res["Fy"]
        d_phys = max(float(np.asarray(d).ravel()[0]), 1e-8)

        dML = Config.SIGN_M1 * (180.0 / np.pi) * K[0, :]
        dMR = Config.SIGN_M2 * (180.0 / np.pi) * K[1, :]
        dFx = Config.SIGN_FX * K[2, :]

        dFy = (dMR - dML) / d_phys
        dFy[2] -= Fy / d_phys                                   # quotient-rule correction for ∂/∂d

        J = np.stack([dML, dMR, dFx, dFy])
        return {
            "J":           J,
            "load_names":  ["ML", "MR", "Fx", "Fy"],
            "input_names": ["phi1 (deg)", "phi2 (deg)", "d"],
        }

    def sensitivity_fd(self, phi1, phi2, d, eps_phi: float = 0.5, eps_d: float = 5e-4):
        """4×3 Jacobian via central finite differences on the model's load outputs.

        Evaluates loads at (x ± δeᵢ) for each input dimension and computes:
            J[k, i] = (loads_k(x + δeᵢ) − loads_k(x − δeᵢ)) / (2δ)

        This is more reliable than sensitivity() because:
        - It uses the well-trained first derivatives (forces) rather than
          unsupervised second derivatives of U.
        - Any high-frequency oscillations in the network are averaged out
          by the finite-difference stencil.

        Args:
            eps_phi: perturbation for angular inputs (degrees). Default 0.5°.
            eps_d:   perturbation for d. Default 5e-4.

        Returns same dict format as sensitivity().
        """
        phi1 = float(phi1)
        phi2 = float(phi2)
        d    = float(d)
        eps  = [eps_phi, eps_phi, eps_d]

        J = np.zeros((4, 3), dtype=np.float64)
        pts = [(phi1, phi2, d)]     # centre — not used in FD but kept for reference

        for col, (dphi1, dphi2, dd) in enumerate([
            (eps[0], 0.0,    0.0   ),   # perturb φ₁
            (0.0,    eps[1], 0.0   ),   # perturb φ₂
            (0.0,    0.0,    eps[2]),   # perturb d
        ]):
            f_plus  = self._loads_at(phi1 + dphi1, phi2 + dphi2, d + dd)
            f_minus = self._loads_at(phi1 - dphi1, phi2 - dphi2, d - dd)
            J[:, col] = (f_plus - f_minus) / (2.0 * eps[col])

        return {
            "J":           J,
            "load_names":  ["ML", "MR", "Fx", "Fy"],
            "input_names": ["phi1 (deg)", "phi2 (deg)", "d"],
        }


def _print_jacobian(label, sens):
    J, loads, inputs = sens["J"], sens["load_names"], sens["input_names"]
    print(f"  {label}:")
    print(f"    {'':8s} {inputs[0]:>14s} {inputs[1]:>14s} {inputs[2]:>10s}")
    for i, ln in enumerate(loads):
        print(f"    {ln:8s} {J[i,0]:14.4e} {J[i,1]:14.4e} {J[i,2]:10.4e}")


if __name__ == "__main__":
    predictor = EnergyPredictor()
    test_cases = [(0.0, 10.0, 0.82), (12.0765, 20.9235, 0.95), (10.0, 0.0, 0.63)]

    for phi1, phi2, d in test_cases:
        print(f"\n{'='*60}")
        print(f"Input: phi1={phi1}°  phi2={phi2}°  d={d}")

        res = predictor.query(phi1, phi2, d, compute_stiffness=False)
        print(f"  Energy={res['Energy']:.6f}  Fx={res['Fx']:.6f}  Fy={res['Fy']:.6f}")
        print(f"  ML={res['M_left']:.6f}  MR={res['M_right']:.6f}")

        _print_jacobian("Sensitivity — autograd (unsupervised 2nd deriv, indicative only)",
                        predictor.sensitivity(phi1, phi2, d))
        _print_jacobian("Sensitivity — finite diff on trained forces (more reliable)",
                        predictor.sensitivity_fd(phi1, phi2, d))
