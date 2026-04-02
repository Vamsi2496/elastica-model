import os, time, json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from config  import Config
from dataset import get_loaders
from model   import ElasticaScalarNet
from loss    import ElasticaLoss


def train():
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    train_loader, val_loader, _, dataset = get_loaders(Config.HDF5_PATH)
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}\n")

    model = ElasticaScalarNet().to(device)
    print(f"Parameters    : {model.count_params():,}\n")

    criterion = ElasticaLoss(dataset)
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.LR,
                            weight_decay=Config.WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer,
                           max_lr          = Config.LR,
                           steps_per_epoch = len(train_loader),
                           epochs          = Config.EPOCHS,
                           pct_start       = 0.1)

    history  = {"train": [], "val": [], "breakdown": []}
    best_val = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y, arc, theta) in enumerate(train_loader):
            x, y, arc, theta = [t.to(device) for t in (x, y, arc, theta)]

            # Model only needs phi — no arc, no theta
            scalar_pred = model(x)

            # Loss uses TRUE theta and arc as physics teachers
            loss, bd = criterion(scalar_pred, y, theta, arc)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           Config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % Config.LOG_INTERVAL == 0:
                print(f"  Ep {epoch:3d} | step {step:3d} | "
                      f"total={bd['total']:.5f} | "
                      f"scalar={bd['scalar']:.5f} | "
                      f"BC={bd['BC_moment']:.5f} | "
                      f"eq={bd['equilibrium']:.5f} | "
                      f"cons={bd['consistency']:.5f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_bd = evaluate(model, val_loader, criterion, device)
        avg_train = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch:3d}/{Config.EPOCHS} | "
              f"train={avg_train:.5f} | val={val_loss:.5f} | "
              f"time={time.time()-t0:.1f}s\n")

        history["train"].append(avg_train)
        history["val"].append(val_loss)
        history["breakdown"].append(val_bd)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss"   : val_loss,
            }, Config.CKPT_BEST)
            print(f"  ✓ Checkpoint saved (val={val_loss:.6f})\n")

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete.")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, bd_sum = 0.0, {}
    for x, y, arc, theta in loader:
        x, y, arc, theta = [t.to(device) for t in (x, y, arc, theta)]
        sp = model(x)                          # scalar only
        loss, bd = criterion(sp, y, theta, arc)
        total += loss.item()
        for k, v in bd.items():
            bd_sum[k] = bd_sum.get(k, 0.0) + v
    n = len(loader)
    return total / n, {k: v / n for k, v in bd_sum.items()}


if __name__ == "__main__":
    train()