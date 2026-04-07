import os
import time
import json
import warnings
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    history = {"train": [], "val": [], "breakdown": []}
    best_val = float("inf")
    no_improve_count = 0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            loss, bd = criterion(model, x, y, need_stiffness=False)

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
                    f"U={bd['energy']:.5f} | "
                    f"Fx={bd['Fx']:.5f} | "
                    f"M1={bd['M_left']:.5f} | "
                    f"M2={bd['M_right']:.5f} | "
                    f"Kreg={bd['stiffness']:.5f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        val_loss, val_bd = evaluate(model, val_loader, criterion)
        avg_train = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        if Config.USE_GPU:
            mem = torch.cuda.memory_reserved(0) / 1e9
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | "
                f"train={avg_train:.5f} | val={val_loss:.5f} | "
                f"time={elapsed:.1f}s | VRAM={mem:.2f}GB")
        else:
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | "
                f"train={avg_train:.5f} | val={val_loss:.5f} | "
                f"time={elapsed:.1f}s")

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


def evaluate(model, loader, criterion):
    model.eval()
    total, bd_sum = 0.0, {}

    for x, y in loader:
        loss, bd = criterion(model, x, y, need_stiffness=False)
        total += loss.item()
        for k, v in bd.items():
            bd_sum[k] = bd_sum.get(k, 0.0) + v

    n = len(loader)
    return total / n, {k: v / n for k, v in bd_sum.items()}


if __name__ == "__main__":
    train()
