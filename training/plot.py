import json
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

with open("training_history.json", "r") as f:
    history = json.load(f)

with open("test_results.json", "r") as f:
    test_results = json.load(f)

train_loss = history["train"]
val_loss = history["val"]
breakdown = history["breakdown"]
epochs = list(range(1, len(train_loss) + 1))
mid = len(epochs) // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(epochs, train_loss, label="Train", color="steelblue", linewidth=2)
ax.plot(epochs, val_loss, label="Val", color="tomato", linewidth=2, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss")
ax.set_title("Train vs Validation Loss")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(epochs[mid:], train_loss[mid:], label="Train", color="steelblue", linewidth=2)
ax2.plot(epochs[mid:], val_loss[mid:], label="Val", color="tomato", linewidth=2, linestyle="--")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Total Loss")
ax2.set_title(f"Zoomed — Last {len(epochs) - mid} Epochs")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_curves.png")

breakdown_keys = {
    "energy": ("Energy", "black"),
    "Fx": ("Fx grad-loss", "steelblue"),
    "M_left": ("M_left grad-loss", "green"),
    "M_right": ("M_right grad-loss", "purple"),
    "stiffness": ("Stiffness reg", "crimson"),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Validation Loss Breakdown", fontsize=14, fontweight="bold")

ax = axes[0]
for key, (label, color) in breakdown_keys.items():
    vals = [bd.get(key, 0.0) for bd in breakdown]
    ax.plot(epochs, vals, label=label, color=color, linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Value")
ax.set_title("All Components (log scale)")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for key, (label, color) in breakdown_keys.items():
    vals = [bd.get(key, 0.0) for bd in breakdown]
    ax2.plot(epochs[mid:], vals[mid:], label=label, color=color, linewidth=1.8)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss Value")
ax2.set_title(f"Zoomed — Last {len(epochs) - mid} Epochs (linear)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_breakdown.png")

names = list(test_results.keys())
r2vals = [test_results[n]["R2"] for n in names]
bar_colors = ["steelblue" if v >= 0.95 else "darkorange" if v >= 0.90 else "tomato" for v in r2vals]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, r2vals, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.5)
ax.axhline(y=0.95, color="green", linestyle="--", linewidth=1.2, label="R²=0.95 target")
ax.axhline(y=0.99, color="blue", linestyle=":", linewidth=1.2, label="R²=0.99 target")
ax.set_ylim(max(0.0, min(r2vals) - 0.05), 1.02)
ax.set_ylabel("R² Score")
ax.set_title("Test R² per Output")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("plots/test_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_r2.png")

scalar_names = list(test_results.keys())
rmse_vals = [test_results[n]["RMSE"] for n in scalar_names]
maxerr_vals = [test_results[n]["MaxErr"] for n in scalar_names]

x = np.arange(len(scalar_names))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, rmse_vals, w, label="RMSE", color="steelblue", edgecolor="black", linewidth=0.8)
b2 = ax.bar(x + w/2, maxerr_vals, w, label="Max Err", color="tomato", edgecolor="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scalar_names)
ax.set_ylabel("Error (physical units)")
ax.set_title("Test RMSE and Max Error per Output")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02 if bar.get_height() != 0 else 0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("plots/test_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_errors.png")

fig, ax = plt.subplots(figsize=(11, 4))
for key, color, label in [
    ("energy", "black", "Energy"),
    ("Fx", "steelblue", "Fx"),
    ("M_left", "green", "M_left"),
    ("M_right", "purple", "M_right"),
    ("stiffness", "crimson", "K-reg"),
]:
    vals = [bd.get(key, 0.0) for bd in breakdown]
    ax.plot(epochs, vals, label=label, color=color, linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Energy / Gradient Loss Convergence")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/gradient_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/gradient_convergence.png")
