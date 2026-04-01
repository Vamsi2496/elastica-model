import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Load training history ─────────────────────────────────────────── #
with open("training_history.json", "r") as f:
    history = json.load(f)

train_loss = history["train"]
val_loss   = history["val"]
breakdown  = history["breakdown"]   # list of dicts, one per epoch

epochs = list(range(1, len(train_loss) + 1))

# ── Load test results ─────────────────────────────────────────────── #
with open("test_results.json", "r") as f:
    test_results = json.load(f)

# ── Load per-node theta RMSE ──────────────────────────────────────── #
node_rmse = np.load("node_rmse.npy")   # (201,)

os.makedirs("plots", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 1 — Training vs Validation Loss
# ═══════════════════════════════════════════════════════════════════ #
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=14, fontweight="bold")

# Left: full loss curve
ax = axes[0]
ax.plot(epochs, train_loss, label="Train", color="steelblue", linewidth=2)
ax.plot(epochs, val_loss,   label="Val",   color="tomato",    linewidth=2,
        linestyle="--")
ax.set_xlabel("Epoch");  ax.set_ylabel("Total Loss")
ax.set_title("Train vs Validation Loss")
ax.legend();  ax.grid(True, alpha=0.3)
ax.set_yscale("log")     # log scale shows early drop clearly

# Right: zoomed in (last 50% of epochs)
ax2 = axes[1]
mid = len(epochs) // 2
ax2.plot(epochs[mid:], train_loss[mid:], label="Train",
         color="steelblue", linewidth=2)
ax2.plot(epochs[mid:], val_loss[mid:],   label="Val",
         color="tomato",    linewidth=2, linestyle="--")
ax2.set_xlabel("Epoch");  ax2.set_ylabel("Total Loss")
ax2.set_title(f"Zoomed — Last {len(epochs)-mid} Epochs")
ax2.legend();  ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_curves.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 2 — Loss Breakdown per Epoch (scalar, theta, phys, cons)
# ═══════════════════════════════════════════════════════════════════ #
keys   = ["scalar", "theta", "BC_moment", "equilibrium", "consistency"]
colors = ["steelblue", "darkorange", "green", "purple", "crimson"]
labels = ["Scalar (Fx,Fy,M)", "Theta field", "BC Moment", "Equilibrium ODE",
          "Consistency"]

fig, ax = plt.subplots(figsize=(12, 5))
for key, color, label in zip(keys, colors, labels):
    vals = [bd.get(key, 0.0) for bd in breakdown]
    ax.plot(epochs, vals, label=label, color=color, linewidth=1.8)

ax.set_xlabel("Epoch");  ax.set_ylabel("Loss Component")
ax.set_title("Validation Loss Breakdown by Component")
ax.set_yscale("log")
ax.legend(loc="upper right");  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_breakdown.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 3 — Test R² Bar Chart
# ═══════════════════════════════════════════════════════════════════ #
names  = list(test_results.keys())                    # Fx,Fy,M_left,M_right,theta
r2vals = [test_results[n]["R2"] for n in names]
colors_bar = ["steelblue" if v > 0.95 else
              "darkorange" if v > 0.90 else
              "tomato" for v in r2vals]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, r2vals, color=colors_bar, edgecolor="black",
              linewidth=0.8, width=0.5)
ax.axhline(y=0.95, color="green",  linestyle="--", linewidth=1.2,
           label="R²=0.95 target")
ax.axhline(y=0.99, color="blue",   linestyle=":",  linewidth=1.2,
           label="R²=0.99 target")
ax.set_ylim(min(r2vals) - 0.05, 1.01)
ax.set_ylabel("R² Score");  ax.set_title("Test R² per Output")
ax.legend();  ax.grid(True, axis="y", alpha=0.3)

# Annotate bars with values
for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("plots/test_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_r2.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 4 — RMSE and MaxErr per scalar output
# ═══════════════════════════════════════════════════════════════════ #
scalar_names = [n for n in test_results if n != "theta"]
rmse_vals    = [test_results[n]["RMSE"]   for n in scalar_names]
maxerr_vals  = [test_results[n]["MaxErr"] for n in scalar_names]

x   = np.arange(len(scalar_names))
w   = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, rmse_vals,   w, label="RMSE",    color="steelblue",
            edgecolor="black", linewidth=0.8)
b2 = ax.bar(x + w/2, maxerr_vals, w, label="Max Err", color="tomato",
            edgecolor="black", linewidth=0.8)

ax.set_xticks(x);  ax.set_xticklabels(scalar_names)
ax.set_ylabel("Error (physical units)")
ax.set_title("Test RMSE and Max Error per Scalar Output")
ax.legend();  ax.grid(True, axis="y", alpha=0.3)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("plots/test_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_errors.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 5 — Per-node Theta RMSE along arc length
# ═══════════════════════════════════════════════════════════════════ #
node_idx = np.arange(len(node_rmse))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(node_idx, node_rmse, color="darkorange", linewidth=1.8)
ax.fill_between(node_idx, node_rmse, alpha=0.2, color="darkorange")
ax.axhline(y=node_rmse.mean(), color="steelblue", linestyle="--",
           linewidth=1.4, label=f"Mean RMSE = {node_rmse.mean():.4f}")
ax.scatter(node_rmse.argmax(), node_rmse.max(), color="red",  zorder=5,
           label=f"Worst node {node_rmse.argmax()} = {node_rmse.max():.4f}")
ax.scatter(node_rmse.argmin(), node_rmse.min(), color="green", zorder=5,
           label=f"Best  node {node_rmse.argmin()} = {node_rmse.min():.4f}")
ax.set_xlabel("Node index (0 = left end, 200 = right end)")
ax.set_ylabel("RMSE of θ (rad)")
ax.set_title("Per-Node Theta RMSE along Arc Length")
ax.legend();  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/theta_node_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/theta_node_rmse.png")


# ═══════════════════════════════════════════════════════════════════ #
print("\nAll plots saved in  plots/")
print("  plots/loss_curves.png")
print("  plots/loss_breakdown.png")
print("  plots/test_r2.png")
print("  plots/test_errors.png")
print("  plots/theta_node_rmse.png")