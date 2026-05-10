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


if __name__ == "__main__":
    predictor = EnergyPredictor()
    test_cases = [(0.0, 10.0, 0.82), (12.0765, 20.9235, 0.95), (10.0, 0.0, 0.63)]
    for phi1, phi2, d in test_cases:
        res = predictor.query(phi1, phi2, d, compute_stiffness=False)
        print(f"\nInput: phi1={phi1}, phi2={phi2}, d={d}")
        print(res)
