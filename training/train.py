import json
import os
import time
import warnings
import numpy as np
import torch
import torch.optim as optim

from config import Config
from dataset import get_loaders
from loss import ElasticaLoss
from model import ElasticaEnergyNet

warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step.*before.*optimizer.step", category=UserWarning)


def evaluate(model, loader, criterion):
    model.eval()
    total, bd_sum = 0.0, {}
    for x, y, arc, theta in loader:
        loss, bd = criterion(model, x, y, arc, theta, need_stiffness=False)
        total += loss.item()
        for k, v in bd.items():
            bd_sum[k] = bd_sum.get(k, 0.0) + float(v)
    n = len(loader)
    return total / n, {k: v / n for k, v in bd_sum.items()}


def print_validation_sample(model, val_loader, dataset, sample_idx=0):
    model.eval()
    x, y, _, _ = next(iter(val_loader))
    x_req = x.detach().requires_grad_(True)
    U = model(x_req)
    g = torch.autograd.grad(U.sum(), x_req, create_graph=False)[0]
    x_phys = x.detach().cpu().numpy() * dataset.x_std[None, :] + dataset.x_mean[None, :]
    d_phys = np.clip(x_phys[:, 2], 1e-8, None)
    scale = dataset.y_std[0] / dataset.x_std
    g_phys = g.detach().cpu().numpy() * scale[None, :]
    U_phys = U.detach().cpu().numpy() * dataset.y_std[0] + dataset.y_mean[0]
    ML_phys = Config.SIGN_M1 * g_phys[:, 0] * (180 / np.pi)
    MR_phys = Config.SIGN_M2 * g_phys[:, 1] * (180 / np.pi)
    Fx_phys = Config.SIGN_FX * g_phys[:, 2]
    Fy_phys = (MR_phys - ML_phys) / d_phys
    auto_phys = y.detach().cpu().numpy() * dataset.y_std[None, :] + dataset.y_mean[None, :]
    i = sample_idx
    print("=== Validation sample ===")
    print(f"input x_phys: {x_phys[i]}")
    print(f"AUTO truth [Energy,Fx,Fy,ML,MR]: {auto_phys[i]}")
    print(f"model pred [Energy,Fx,Fy,ML,MR]: {[U_phys[i], Fx_phys[i], Fy_phys[i], ML_phys[i], MR_phys[i]]}")


def train():
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Architecture: 3 -> {' -> '.join(map(str, Config.HIDDEN_LAYERS))} -> 1")
    if Config.USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, _, dataset = get_loaders(Config.HDF5_PATH)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    model = ElasticaEnergyNet().to(device)
    print(f"Parameters: {model.count_params():,}")
    criterion = ElasticaLoss(dataset)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE, min_lr=Config.MIN_LR, threshold=Config.LR_THRESHOLD)
    history = {"train": [], "val": [], "breakdown": []}
    best_val = float("inf")
    no_improve_count = 0
    start_epoch = 1

    if os.path.exists(Config.CKPT_LATEST):
        ckpt = torch.load(Config.CKPT_LATEST, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        no_improve_count = ckpt["no_improve_count"]
        history = ckpt["history"]
        print(f"Resumed from epoch {ckpt['epoch']} (best_val={best_val:.6f}, no_improve={no_improve_count})")

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        # curriculum: linearly ramp energy weight down and moment weight up
        if epoch <= Config.CURRICULUM_EPOCHS:
            frac = epoch / Config.CURRICULUM_EPOCHS
            Config.W_ENERGY_LABEL = Config.W_ENERGY_LABEL_INIT + frac * (20.0 - Config.W_ENERGY_LABEL_INIT)
            Config.M_WEIGHT = Config.M_WEIGHT_INIT + frac * (10.0 - Config.M_WEIGHT_INIT)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, (x, y, arc, theta) in enumerate(train_loader):
            loss, bd = criterion(model, x, y, arc, theta, need_stiffness=False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            if step % Config.LOG_INTERVAL == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Ep {epoch:3d} | step {step:4d} | total={bd['total']:.5f} | energy={bd['energy']:.5f} | scalar={bd['scalar']:.5f} | Kreg={bd['stiffness']:.5f} | lr={lr:.2e}")
        val_loss, val_bd = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        print_validation_sample(model, val_loader, dataset, sample_idx=0)
        avg_train = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        curriculum_tag = f" | W_E={Config.W_ENERGY_LABEL:.1f} W_M={Config.M_WEIGHT:.1f}" if epoch <= Config.CURRICULUM_EPOCHS else ""
        if Config.USE_GPU:
            mem = torch.cuda.memory_reserved(0) / 1e9
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | train={avg_train:.5f} | val={val_loss:.5f} | time={elapsed:.1f}s | VRAM={mem:.2f}GB{curriculum_tag}")
        else:
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | train={avg_train:.5f} | val={val_loss:.5f} | time={elapsed:.1f}s{curriculum_tag}")
        history["train"].append(avg_train)
        history["val"].append(val_loss)
        history["breakdown"].append(val_bd)
        if val_loss < best_val - Config.MIN_DELTA:
            best_val = val_loss
            no_improve_count = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "val_loss": val_loss, "architecture": Config.HIDDEN_LAYERS}, Config.CKPT_BEST)
            print(f"✓ Checkpoint saved (val={val_loss:.6f})")
        else:
            no_improve_count += 1
            print(f"- No improvement for {no_improve_count}/{Config.PATIENCE} epoch(s) (best val={best_val:.6f})")
        with open("training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val": best_val,
            "no_improve_count": no_improve_count,
            "history": history,
        }, Config.CKPT_LATEST)
        if no_improve_count >= Config.PATIENCE:
            print(f"⏹ Early stopping at epoch {epoch}")
            break
    print(f"Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    train()