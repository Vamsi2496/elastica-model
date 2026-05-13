import json
import numpy as np
import torch
import time

from config import Config
from dataset import get_loaders
from model import ElasticaEnergyNet


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def test():
    start = time.time()
    device = Config.DEVICE
    print(f"Device: {device}")
    ckpt = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaEnergyNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.6f}")
    print(f"d-slice: {Config.D_SLICE} ± {Config.D_SLICE_TOL}")
    print(f"Architecture: MLP  2 -> {' -> '.join(map(str, Config.HIDDEN_LAYERS))} -> 1  (GELU)")
    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH, compute_stats=False)

    pred_all, true_auto, x_all = [], [], []
    for x, y in test_loader:
        x_req = x.detach().requires_grad_(True)
        U = model(x_req)
        g = torch.autograd.grad(U.sum(), x_req, create_graph=False)[0]
        x_phys = x.detach().cpu().numpy() * dataset.x_std[None, :] + dataset.x_mean[None, :]
        scale = dataset.y_std[0] / dataset.x_std
        g_phys = g.detach().cpu().numpy() * scale[None, :]
        U_phys = U.detach().cpu().numpy() * dataset.y_std[0] + dataset.y_mean[0]
        ML_phys = Config.SIGN_M1 * g_phys[:, 0] * (180 / np.pi)
        MR_phys = Config.SIGN_M2 * g_phys[:, 1] * (180 / np.pi)
        pred_all.append(np.stack([U_phys, ML_phys, MR_phys], axis=1))
        true_auto.append(y.detach().cpu().numpy() * dataset.y_std[None, :] + dataset.y_mean[None, :])
        x_all.append(x_phys)

    pred_all  = np.concatenate(pred_all)
    true_auto = np.concatenate(true_auto)
    x_all     = np.concatenate(x_all)

    print("=" * 60)
    print(f"{'Output':<12} {'R²':>9} {'RMSE':>12} {'MaxErr':>12}")
    print("=" * 60)
    results = {}
    for i, name in enumerate(Config.SCALAR_NAMES):
        r2_v    = r2(true_auto[:, i], pred_all[:, i])
        rmse_v  = np.sqrt(np.mean((true_auto[:, i] - pred_all[:, i]) ** 2))
        maxerr  = np.max(np.abs(true_auto[:, i] - pred_all[:, i]))
        print(f"{name:<12} {r2_v:>9.5f} {rmse_v:>12.4e} {maxerr:>12.4e}")
        results[name] = {"R2": float(r2_v), "RMSE": float(rmse_v), "MaxErr": float(maxerr)}
    print("=" * 60)
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → test_results.json")

    # --- outlier diagnostics ---
    abs_err = np.abs(true_auto - pred_all)
    max_err_per_sample = abs_err.max(axis=1)
    top_k = 2000
    worst_idx = np.argsort(max_err_per_sample)[-top_k:]

    print(f"\nTop-{top_k} outlier samples (worst 20 shown):")
    pad = " " * 24
    header = f"  {'phi1':>7}  {'phi2':>7}  | {'Output':<8} {'True':>10} {'Pred':>10} {'AbsErr':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in worst_idx[-20:][::-1]:
        for j, name in enumerate(Config.SCALAR_NAMES):
            prefix = f"  {x_all[i,0]:7.3f}  {x_all[i,1]:7.3f}" if j == 0 else pad
            print(f"{prefix}  | {name:<8} {true_auto[i,j]:10.4f} {pred_all[i,j]:10.4f} {abs_err[i,j]:10.4f}")
        print()

    np.savez("outliers.npz",
             phi1=x_all[worst_idx, 0],
             phi2=x_all[worst_idx, 1],
             true=true_auto[worst_idx],
             pred=pred_all[worst_idx],
             abs_err=abs_err[worst_idx],
             output_names=np.array(Config.SCALAR_NAMES))
    print("Outlier data saved → outliers.npz")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    test()
