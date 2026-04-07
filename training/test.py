import torch
import numpy as np
import json

from config import Config
from dataset import get_loaders
from model import ElasticaEnergyNet


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def to_physical_grads(g_norm, dataset):
    u_std = dataset.y_std[0]
    scales = u_std / dataset.x_std
    out = g_norm * scales[None, :]
    return out


def hessian_to_physical(H_norm, dataset):
    u_std = dataset.y_std[0]
    scales = u_std / dataset.x_std
    S = scales[:, None] * scales[None, :]
    return H_norm * S[None, :, :]


def test():
    device = Config.DEVICE
    print(f"Device : {device}")

    ckpt = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaEnergyNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.6f}")

    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH, compute_stats=False)
    y_mean = dataset.y_mean
    y_std = dataset.y_std

    pred_all, true_all = [], []

    for x, y in test_loader:
        x_req = x.detach().requires_grad_(True)
        U = model(x_req)
        g = torch.autograd.grad(U.sum(), x_req, create_graph=False)[0]

        fx_phys = Config.SIGN_FX * to_physical_grads(g.detach().cpu().numpy(), dataset)[:, 2]
        m1_phys = Config.SIGN_M1 * to_physical_grads(g.detach().cpu().numpy(), dataset)[:, 0]
        m2_phys = Config.SIGN_M2 * to_physical_grads(g.detach().cpu().numpy(), dataset)[:, 1]
        U_phys = (U.detach().cpu().numpy() * y_std[0]) + y_mean[0]

        pred = np.stack([U_phys, fx_phys, m1_phys, m2_phys], axis=1)
        true = (y.detach().cpu().numpy() * y_std[None, :]) + y_mean[None, :]

        pred_all.append(pred)
        true_all.append(true)

    SP = np.concatenate(pred_all)
    ST = np.concatenate(true_all)

    print("=" * 72)
    print(f"{'Output':<12} {'R²':>9} {'RMSE':>13} {'MaxErr':>13}")
    print("=" * 72)

    results = {}
    for i, name in enumerate(Config.SCALAR_NAMES):
        r2_val = r2(ST[:, i], SP[:, i])
        rmse = np.sqrt(np.mean((ST[:, i] - SP[:, i]) ** 2))
        maxerr = np.max(np.abs(ST[:, i] - SP[:, i]))
        print(f"{name:<12} {r2_val:>9.5f} {rmse:>13.4e} {maxerr:>13.4e}")
        results[name] = {"R2": float(r2_val), "RMSE": float(rmse), "MaxErr": float(maxerr)}

    print("=" * 72)
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → test_results.json")


if __name__ == "__main__":
    test()
