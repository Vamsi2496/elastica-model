import torch
import numpy as np
from config import Config
from dataset import get_loaders
from model import ElasticaEnergyNet
from loss import ElasticaLoss


def diagnose_signs(ckpt_path=Config.CKPT_BEST):
    device = Config.DEVICE
    _, _, test_loader, dataset = get_loaders(Config.HDF5_PATH, compute_stats=False)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ElasticaEnergyNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    criterion = ElasticaLoss(dataset)
    
    batch = next(iter(test_loader))
    x, y_auto, arc, theta = [t.to(device) for t in batch]
    
    # Predicted energy + gradients
    x_req = x.detach().requires_grad_(True)
    U_pred = model(x_req)
    g = torch.autograd.grad(U_pred.sum(), x_req, create_graph=False)[0]
    
    # Physical predictions
    x_phys = criterion._dn_x(x, device)
    d_phys = x_phys[:, 2].clamp(min=1e-8)
    
    scale = dataset.y_std[0] / dataset.x_std
    scale_t = torch.from_numpy(scale).to(device)    
    g_phys = g * scale_t[None, :]
    
    ML_phys = Config.SIGN_M1 * g_phys[:, 0]
    MR_phys = Config.SIGN_M2 * g_phys[:, 1]
    Fx_phys = Config.SIGN_FX * g_phys[:, 2]
    Fy_phys = (ML_phys - MR_phys) / d_phys  # Your corrected convention
    
    U_phys = (U_pred * dataset.y_std[0]) + dataset.y_mean[0]
    
    # Denormalize AUTO targets
    y_auto_phys = (y_auto * dataset.y_std[None, :]) + dataset.y_mean[None, :]
    
    # Sign correlations using dictionary instead of locals()
    results = {
        "Energy": U_phys,
        "Fx": Fx_phys,
        "Fy": Fy_phys,
        "M_left": ML_phys,
        "M_right": MR_phys
    }
    
    names = Config.SCALAR_NAMES
    print("═" * 80)
    print("SIGN CORRELATION DIAGNOSTIC (1 batch)")
    print("═" * 80)
    print(f"{'Target':<12} {'Pred mean':>12} {'AUTO mean':>12} {'Sign corr':>12} {'Sign match %':>12}")
    print("─" * 80)
    
    for i, name in enumerate(names):
        pred = results[name]
        auto = y_auto_phys[:, i]
        pred_mean = pred.mean().item()
        auto_mean = auto.mean().item()
        sign_corr = (torch.sign(pred) * torch.sign(auto) > 0).float().mean().item()
        sign_match = (torch.sign(pred) * auto > 0).float().mean().item()
        
        print(f"{name:<12} {pred_mean:>12.4f} {auto_mean:>12.4f} {sign_corr:>12.3f} {sign_match:>12.1f}%")
    
    # Normalization stats
    print("\n═" * 80)
    print("NORMALIZATION STATS")
    print("═" * 80)
    print(f"{'Target':<12} {'y_mean':>12} {'y_std':>12} {'Ulab vs mean':>12}")
    print("─" * 80)
    Ulab_mean = y_auto_phys[:, 0].mean().item()
    for i, name in enumerate(names):
        print(f"{name:<12} {dataset.y_mean[i]:>12.4f} {dataset.y_std[i]:>12.4f} {Ulab_mean:>12.4f}")
    
    # Recommended signs
    print("\n═" * 80)
    print("RECOMMENDED SIGN FLIPS")
    print("═" * 80)
    for i, name in enumerate(names):
        pred = results[name]
        auto = y_auto_phys[:, i]
        sign_match = (torch.sign(pred) * auto > 0).float().mean().item()
        status = "FLIP NOW" if sign_match < 0.6 else "FLIP MAYBE" if sign_match < 0.8 else "OK"
        print(f"{name}: {status} ({sign_match:.1%} match)")
    
    print("\nRun this to get exact sign recommendations!")


if __name__ == "__main__":
    diagnose_signs()