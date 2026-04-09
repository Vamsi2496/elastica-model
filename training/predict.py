import numpy as np
import torch
from model  import ElasticaScalarNet
from config import Config


def load_model(ckpt_path: str, device):
    model = ElasticaScalarNet().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    return model


def predict(model, phi1, phi2, d, stats, device):
    """
    phi1, phi2, d : scalars or 1-D arrays of the same length
    Returns dict with Fx, Fy, M_left, M_right as numpy arrays.
    """
    x = np.stack([
        np.atleast_1d(np.float32(phi1)),
        np.atleast_1d(np.float32(phi2)),
        np.atleast_1d(np.float32(d)),
    ], axis=1)                                               # (N, 3)

    x = (x - stats["x_mean"]) / stats["x_std"]             # normalise

    with torch.no_grad():
        x_t  = torch.from_numpy(x).to(device)               # (N, 3)
        y_t  = model(x_t)                                    # (N, 4) normalised
        y    = y_t.cpu().numpy()

    y = y * stats["y_std"] + stats["y_mean"]                # denormalise

    return {
        "Fx"    : y[:, 0],
        "Fy"    : y[:, 1],
        "M_left": y[:, 2],
        "M_right": y[:, 3],
    }


if __name__ == "__main__":
    device = Config.DEVICE

    # 1. Load norm stats
    stats = np.load(Config.NORM_STATS)

    # 2. Load trained model
    model = load_model(Config.CKPT_BEST, device)

    # ── Single sample ─────────────────────────────────────────────── #
    result = predict(model, phi1=0, phi2=0, d=0.90, stats=stats, device=device)
    print("\nSingle prediction:")
    for k, v in result.items():
        print(f"  {k:8s} = {v[0]:.4f}")

    # ── Batch of samples ──────────────────────────────────────────── #
    phi1_arr = np.array([0, 0, -0], dtype=np.float32)
    phi2_arr = np.array([0, 0, -0], dtype=np.float32)
    d_arr    = np.array([0.99, 0.80, 0.60], dtype=np.float32)

    result_batch = predict(model, phi1_arr, phi2_arr, d_arr, stats=stats, device=device)
    print("\nBatch predictions:")
    print(f"  {'Sample':>6}  {'Fx':>10}  {'Fy':>10}  {'M_left':>10}  {'M_right':>10}")
    for i in range(len(phi1_arr)):
        print(f"  {i:>6}  "
              f"{result_batch['Fx'][i]:>10.4f}  "
              f"{result_batch['Fy'][i]:>10.4f}  "
              f"{result_batch['M_left'][i]:>10.4f}  "
              f"{result_batch['M_right'][i]:>10.4f}")