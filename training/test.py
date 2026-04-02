import torch
import numpy as np
import json

from config  import Config
from dataset import get_loaders
from model   import ElasticaScalarNet


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaScalarNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}\n")

    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH,
                                             compute_stats=False)
    y_mean = torch.from_numpy(dataset.y_mean).to(device)
    y_std  = torch.from_numpy(dataset.y_std).to(device)

    sp_all, st_all = [], []

    for x, y, arc, theta in test_loader:
        x, y = x.to(device), y.to(device)
        sp = model(x)                          # only phi needed
        sp_all.append((sp * y_std + y_mean).cpu().numpy())
        st_all.append((y  * y_std + y_mean).cpu().numpy())

    SP = np.concatenate(sp_all)
    ST = np.concatenate(st_all)

    print("=" * 62)
    print(f"{'Output':<12} {'R²':>9} {'RMSE':>13} {'MaxErr':>13}")
    print("=" * 62)

    results = {}
    for i, name in enumerate(Config.SCALAR_NAMES):
        r2_val = r2(ST[:, i], SP[:, i])
        rmse   = np.sqrt(np.mean((ST[:, i] - SP[:, i]) ** 2))
        maxerr = np.max(np.abs(ST[:, i] - SP[:, i]))
        print(f"{name:<12} {r2_val:>9.5f} {rmse:>13.4e} {maxerr:>13.4e}")
        results[name] = {"R2": float(r2_val),
                         "RMSE": float(rmse),
                         "MaxErr": float(maxerr)}

    print("=" * 62)
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → test_results.json")


def predict_single(phi1, phi2, d):
    """
    Predict Fx, Fy, ML, MR for one beam configuration.
    Only phi1, phi2, d needed — no arc length, no theta.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats  = np.load(Config.NORM_STATS)
    x_mean = stats["x_mean"];  x_std = stats["x_std"]
    y_mean = stats["y_mean"];  y_std = stats["y_std"]

    phi   = (np.array([phi1, phi2, d], dtype=np.float32) - x_mean) / x_std
    phi_t = torch.from_numpy(phi).unsqueeze(0).to(device)

    ckpt  = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaScalarNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        sp = model(phi_t)

    sp_phys = sp.cpu().numpy()[0] * y_std + y_mean
    return {"Fx": float(sp_phys[0]), "Fy": float(sp_phys[1]),
            "M_left": float(sp_phys[2]), "M_right": float(sp_phys[3])}


if __name__ == "__main__":
    test()

    print("\n── Single prediction example ──")
    r = predict_single(phi1=0, phi2=0, d=0.91)
    print(f"Fx={r['Fx']:.4f}  Fy={r['Fy']:.6f}  "
          f"ML={r['M_left']:.4f}  MR={r['M_right']:.4f}")