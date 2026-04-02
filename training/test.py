import torch
import numpy as np
import json
from torch.amp import autocast

from config  import Config
from dataset import get_loaders
from model   import ElasticaScalarNet


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def test():
    device = Config.DEVICE
    print(f"Device : {device}\n")

    ckpt  = torch.load(Config.CKPT_BEST, map_location=device)
    model = ElasticaScalarNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if Config.COMPILE:
        model = torch.compile(model)

    print(f"Loaded epoch {ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.6f}\n")

    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH,
                                             compute_stats=False)
    y_mean = torch.tensor(dataset.y_mean, device=device)
    y_std  = torch.tensor(dataset.y_std,  device=device)

    sp_all, st_all = [], []

    for x, y, arc, theta in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type=str(device),
                      enabled=Config.MIXED_PREC):
            sp = model(x)

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
        results[name] = {"R2"    : float(r2_val),
                         "RMSE"  : float(rmse),
                         "MaxErr": float(maxerr)}

    print("=" * 62)
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → test_results.json")


if __name__ == "__main__":
    test()