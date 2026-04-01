import torch
import numpy as np
import json
from torch.amp import autocast

from config  import Config
from dataset import get_loaders
from model   import ElasticaINR


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────── #
    ckpt  = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaINR().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}\n")

    # ── Test loader ───────────────────────────────────────────────────── #
    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH,
                                             compute_stats=False)
    y_mean  = torch.from_numpy(dataset.y_mean).to(device)
    y_std   = torch.from_numpy(dataset.y_std).to(device)
    t_mean  = float(dataset.t_mean)
    t_std   = float(dataset.t_std)
    arc_max = float(dataset.arc_max)

    # ── Collect predictions ───────────────────────────────────────────── #
    sp_all, st_all = [], []
    tp_all, tt_all = [], []

    for x, y, arc, theta in test_loader:
        x, y, arc, theta = [t.to(device) for t in (x, y, arc, theta)]

        with autocast(device_type="cpu", enabled=False):
            sp, tp = model(x, arc)

        # Denormalise scalars
        sp_phys = (sp * y_std + y_mean).cpu().numpy()
        st_phys = (y  * y_std + y_mean).cpu().numpy()

        # Denormalise theta
        tp_phys = tp.cpu().numpy() * t_std + t_mean
        tt_phys = theta.cpu().numpy() * t_std + t_mean

        sp_all.append(sp_phys);  st_all.append(st_phys)
        tp_all.append(tp_phys);  tt_all.append(tt_phys)

    SP = np.concatenate(sp_all, axis=0)    # (N_test, 4)
    ST = np.concatenate(st_all, axis=0)
    TP = np.concatenate(tp_all, axis=0)    # (N_test, 201)
    TT = np.concatenate(tt_all, axis=0)

    # ── Scalar metrics ────────────────────────────────────────────────── #
    print("=" * 62)
    print(f"{'Output':<12} {'R²':>9} {'RMSE':>13} {'MaxErr':>13}")
    print("=" * 62)

    results = {}
    for i, name in enumerate(Config.SCALAR_NAMES):
        r2_val = r2(ST[:, i], SP[:, i])
        rmse   = np.sqrt(np.mean((ST[:, i] - SP[:, i]) ** 2))
        maxerr = np.max(np.abs(ST[:, i] - SP[:, i]))
        print(f"{name:<12} {r2_val:>9.5f} {rmse:>13.4e} {maxerr:>13.4e}")
        results[name] = {
            "R2"    : float(r2_val),
            "RMSE"  : float(rmse),
            "MaxErr": float(maxerr)
        }

    # ── Theta field metrics ───────────────────────────────────────────── #
    r2_theta   = float(np.mean([r2(TT[:, j], TP[:, j])
                                for j in range(Config.N_NODES)]))
    rmse_theta = float(np.sqrt(np.mean((TT - TP) ** 2)))
    node_rmse  = np.sqrt(np.mean((TT - TP) ** 2, axis=0))   # (201,)

    print(f"{'theta[201]':<12} {r2_theta:>9.5f} {rmse_theta:>13.4e}")
    print("=" * 62)
    print(f"Worst node RMSE : {node_rmse.max():.4e}  at node {node_rmse.argmax()}")
    print(f"Best  node RMSE : {node_rmse.min():.4e}  at node {node_rmse.argmin()}")

    results["theta"] = {
        "R2"  : r2_theta,
        "RMSE": rmse_theta
    }

    # ── Save outputs ──────────────────────────────────────────────────── #
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save("node_rmse.npy", node_rmse)
    print("\nSaved → test_results.json,  node_rmse.npy")


# ── Single prediction utility ─────────────────────────────────────── #
def predict_single(phi1, phi2, d,
                   s_query   = None,
                   ckpt_path = Config.CKPT_BEST,
                   norm_path = Config.NORM_STATS):
    """
    Predict forces + θ(s) for one configuration.
    phi1, phi2 : floats in degrees (matches your dataset)
    d          : float
    s_query    : 1-D array of arc length positions (physical units)
                 If None, uses 201 uniform points in [0, arc_max]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats  = np.load(norm_path)

    x_mean  = stats["x_mean"];  x_std  = stats["x_std"]
    y_mean  = stats["y_mean"];  y_std  = stats["y_std"]
    t_mean  = float(stats["t_mean"])
    t_std   = float(stats["t_std"])
    arc_max = float(stats["arc_max"])

    if s_query is None:
        s_query = np.linspace(0, arc_max, Config.N_NODES)

    # Normalise inputs
    phi      = np.array([phi1, phi2, d], dtype=np.float32)
    phi      = (phi - x_mean) / x_std
    arc_norm = (s_query / arc_max).astype(np.float32)

    phi_t = torch.from_numpy(phi).unsqueeze(0).to(device)       # (1, 3)
    arc_t = torch.from_numpy(arc_norm).unsqueeze(0).to(device)  # (1, N)

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ElasticaINR().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        sp, tp = model(phi_t, arc_t)

    sp_phys = sp.cpu().numpy()[0] * y_std + y_mean
    tp_phys = tp.cpu().numpy()[0] * t_std + t_mean

    return {
        "Fx"     : float(sp_phys[0]),
        "Fy"     : float(sp_phys[1]),
        "M_left" : float(sp_phys[2]),
        "M_right": float(sp_phys[3]),
        "theta"  : tp_phys,           # numpy array (N,)
        "arc"    : s_query,           # numpy array (N,)
    }


if __name__ == "__main__":
    test()

    # ── Example single prediction ─────────────────────────────────────── #
    print("\n── Single prediction example ──")
    result = predict_single(phi1=0, phi2=0, d=0.6)
    print(f"Fx      = {result['Fx']:.4f}")
    print(f"Fy      = {result['Fy']:.6f}")
    print(f"M_left  = {result['M_left']:.4f}")
    print(f"M_right = {result['M_right']:.4f}")
    print(f"theta   = {result['theta'][:50]}  ...")