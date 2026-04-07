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

    def _norm_x(self, phi1, phi2, d):
        x = np.stack([
            np.asarray(phi1, np.float32).ravel(),
            np.asarray(phi2, np.float32).ravel(),
            np.asarray(d, np.float32).ravel(),
        ], axis=1)
        return torch.from_numpy((x - self.x_mean) / self.x_std).to(self.device)

    def _grad_to_phys(self, g):
        scale = self.y_std[0] / self.x_std
        return g * scale[None, :]

    def _hess_to_phys(self, H):
        s = self.y_std[0] / self.x_std
        S = s[:, None] * s[None, :]
        return H * S[None, :, :]

    def query(self, phi1, phi2, d, compute_stiffness=True):
        x = self._norm_x(phi1, phi2, d).detach().requires_grad_(True)
        U = self.model(x)
        g = torch.autograd.grad(U.sum(), x, create_graph=compute_stiffness)[0]

        U_phys = (U.detach().cpu().numpy() * self.y_std[0]) + self.y_mean[0]
        g_phys = self._grad_to_phys(g.detach().cpu().numpy())

        out = {
            "Energy": float(U_phys[0]),
            "Fx": float(Config.SIGN_FX * g_phys[0, 2]),
            "M_left": float(Config.SIGN_M1 * g_phys[0, 0]),
            "M_right": float(Config.SIGN_M2 * g_phys[0, 1]),
        }

        if compute_stiffness:
            H = np.zeros((1, 3, 3), dtype=np.float32)
            for i in range(3):
                row = torch.autograd.grad(g[:, i].sum(), x, retain_graph=(i < 2))[0]
                H[:, i, :] = row.detach().cpu().numpy()
            H_phys = self._hess_to_phys(H)[0]
            out["K"] = H_phys
        return out

predictor = EnergyPredictor()

# input values
phi1 = 0
phi2 = 0
d = 0.982

# query
result = predictor.query(phi1, phi2, d)

print(result)
Fy=(result["M_right"]-result["M_left"])/d
print(Fy)