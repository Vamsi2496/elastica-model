import os
import time
import json
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from config import Config
from dataset import get_loaders
from model import ElasticaEnergyNet
from loss import ElasticaLoss

warnings.filterwarnings(
    "ignore",
    message="Detected call of.*lr_scheduler.step.*before.*optimizer.step",
    category=UserWarning,
)


def train():
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    device = Config.DEVICE
    print(f"Device : {device}")

    if Config.USE_GPU:
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _, dataset = get_loaders(Config.HDF5_PATH)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches : {len(val_loader)}")

    model = ElasticaEnergyNet().to(device)
    print(f"Parameters : {model.count_params():,}")

    criterion = ElasticaLoss(dataset)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=Config.LR, steps_per_epoch=len(train_loader), epochs=Config.EPOCHS, pct_start=0.1)

    history = {"train": [], "val": [], "breakdown": []}
    best_val = float("inf")
    no_improve_count = 0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y, arc, theta) in enumerate(train_loader):
            loss, bd = criterion(model, x, y, arc, theta, need_stiffness=False)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % Config.LOG_INTERVAL == 0:
                print(
                    f" Ep {epoch:3d} | step {step:4d} | "
                    f"total={bd['total']:.5f} | "
                    f"Ulab={bd['energy_label']:.5f} | "
                    f"Utheta={bd['energy_theta']:.5f} | "
                    f"scalar={bd['scalar']:.5f} | "
                    f"lstsq={bd['lstsq']:.5f} | "
                    f"eq={bd['equilibrium']:.5f} | "
                    f"Kreg={bd['stiffness']:.5f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        val_loss, val_bd = evaluate(model, val_loader, criterion)
        print_validation_sample(model, val_loader, dataset, sample_idx=0)
        avg_train = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        if Config.USE_GPU:
            mem = torch.cuda.memory_reserved(0) / 1e9
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | train={avg_train:.5f} | val={val_loss:.5f} | time={elapsed:.1f}s | VRAM={mem:.2f}GB")
        else:
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | train={avg_train:.5f} | val={val_loss:.5f} | time={elapsed:.1f}s")

        history["train"].append(avg_train)
        history["val"].append(val_loss)
        history["breakdown"].append(val_bd)

        if val_loss < best_val - Config.MIN_DELTA:
            best_val = val_loss
            no_improve_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                Config.CKPT_BEST,
            )
            print(f" ✓ Checkpoint saved (val={val_loss:.6f})")
        else:
            no_improve_count += 1
            print(f" – No improvement for {no_improve_count}/{Config.PATIENCE} epoch(s) (best val={best_val:.6f})")

        if no_improve_count >= Config.PATIENCE:
            print(f" ⏹ Early stopping at epoch {epoch} (no improvement for {Config.PATIENCE} consecutive epochs)")
            break

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val loss: {best_val:.6f}")


def print_validation_sample(model, val_loader, dataset, sample_idx=0):
    model.eval()
    x, y, arc, theta = next(iter(val_loader))

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

    auto_phys = (y.detach().cpu().numpy() * dataset.y_std[None, :]) + dataset.y_mean[None, :]

    i = sample_idx
    print("=== Validation sample ===")
    print(f"input x_phys           : {x_phys[i]}")
    print(f"AUTO truth [Energy,Fx,Fy,ML,MR]: {auto_phys[i]}")
    print(f"model pred [Energy,Fx,Fy,ML,MR]: {[U_phys[i], Fx_phys[i], Fy_phys[i], ML_phys[i], MR_phys[i]]}")


def evaluate(model, loader, criterion):
    model.eval()
    total, bd_sum = 0.0, {}

    for x, y, arc, theta in loader:
        loss, bd = criterion(model, x, y, arc, theta, need_stiffness=False)
        total += loss.item()
        for k, v in bd.items():
            bd_sum[k] = bd_sum.get(k, 0.0) + v

    n = len(loader)
    return total / n, {k: v / n for k, v in bd_sum.items()}


if __name__ == "__main__":
    train()
