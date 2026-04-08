import torch
import numpy as np
import json

from config import Config
from dataset import get_loaders
from model import ElasticaEnergyNet
from loss import ElasticaLoss


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def test():
    device = Config.DEVICE
    print(f"Device : {device}")

    ckpt = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaEnergyNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.6f}")

    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH, compute_stats=False)

    pred_all, true_auto, true_theta = [], [], []

    for x, y, arc, theta in test_loader:
        x_req = x.detach().requires_grad_(True)
        U = model(x_req)
        g = torch.autograd.grad(U.sum(), x_req, create_graph=False)[0]

        x_phys = x.detach().cpu().numpy() * dataset.x_std[None, :] + dataset.x_mean[None, :]
        d_phys = np.clip(x_phys[:, 2], 1e-8, None)
        scale = dataset.y_std[0] / dataset.x_std
        g_phys = g.detach().cpu().numpy() * scale[None, :]

        U_phys = (U.detach().cpu().numpy() * dataset.y_std[0]) + dataset.y_mean[0]
        ML_phys = Config.SIGN_M1 * g_phys[:, 0]
        MR_phys = Config.SIGN_M2 * g_phys[:, 1]
        Fx_phys = Config.SIGN_FX * g_phys[:, 2]
        Fy_phys = (ML_phys - MR_phys) / d_phys

        pred_all.append(np.stack([U_phys, Fx_phys, Fy_phys, ML_phys, MR_phys], axis=1))
        true_auto.append((y.detach().cpu().numpy() * dataset.y_std[None, :]) + dataset.y_mean[None, :])

        theta_phys = theta * dataset.t_std + dataset.t_mean
        arc_phys = arc * dataset.arc_max
        Fx_t, Fy_t, ML_t, MR_t, _, _ = ElasticaLoss.derive_from_theta(theta_phys, arc_phys)
        U_t = ElasticaLoss.energy_from_theta(theta_phys, arc_phys)
        true_theta.append(torch.stack([U_t, Fx_t, Fy_t, ML_t, MR_t], dim=1).detach().cpu().numpy())

    pred_all = np.concatenate(pred_all)
    true_auto = np.concatenate(true_auto)
    true_theta = np.concatenate(true_theta)

    print("=" * 110)
    print(f"{'Output':<12} {'AUTO R²':>9} {'AUTO RMSE':>12} {'AUTO MaxErr':>12} {'Θ/arc R²':>9} {'Θ/arc RMSE':>12} {'Θ/arc MaxErr':>12}")
    print("=" * 110)

    results = {"AUTO": {}, "theta_arc": {}}
    for i, name in enumerate(Config.SCALAR_NAMES):
        r2_a = r2(true_auto[:, i], pred_all[:, i])
        rmse_a = np.sqrt(np.mean((true_auto[:, i] - pred_all[:, i]) ** 2))
        maxerr_a = np.max(np.abs(true_auto[:, i] - pred_all[:, i]))

        r2_t = r2(true_theta[:, i], pred_all[:, i])
        rmse_t = np.sqrt(np.mean((true_theta[:, i] - pred_all[:, i]) ** 2))
        maxerr_t = np.max(np.abs(true_theta[:, i] - pred_all[:, i]))

        print(f"{name:<12} {r2_a:>9.5f} {rmse_a:>12.4e} {maxerr_a:>12.4e} {r2_t:>9.5f} {rmse_t:>12.4e} {maxerr_t:>12.4e}")
        results["AUTO"][name] = {"R2": float(r2_a), "RMSE": float(rmse_a), "MaxErr": float(maxerr_a)}
        results["theta_arc"][name] = {"R2": float(r2_t), "RMSE": float(rmse_t), "MaxErr": float(maxerr_t)}

    print("=" * 110)
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → test_results.json")


if __name__ == "__main__":
    test()
