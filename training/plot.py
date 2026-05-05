import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))
from config import Config

os.makedirs("plots", exist_ok=True)

with open("training_history.json", "r") as f:
    history = json.load(f)

with open("test_results.json", "r") as f:
    test_results = json.load(f)

train_loss = history.get("train", [])
val_loss   = history.get("val", [])
breakdown  = history.get("breakdown", [])
epochs     = list(range(1, len(train_loss) + 1))

if len(epochs) == 0:
    raise ValueError("training_history.json contains no epochs.")

mid = max(1, len(epochs) // 2)

# ------------------------------------------------------------------
# Which breakdown components are active (weight > 0 in current config)?
# Components whose weight is exactly 0 are not plotted.
# ------------------------------------------------------------------
W = Config.W_SCALAR  # scalar sub-loss multiplier

COMPONENT_WEIGHTS = {
    "energy":       Config.W_ENERGY_LABEL,
    "energy_theta": Config.W_ENERGY_THETA,
    "Fx":           W * Config.FX_WEIGHT,
    "Fx_L4":        W * Config.FX_L4_WEIGHT,
    "Fy":           W * Config.FY_WEIGHT,
    "M_left":       W * Config.M_WEIGHT,
    "M_right":      W * Config.M_WEIGHT,
    "scalar":       W,           # aggregate of all scalar sub-losses
    "stiffness":    Config.LAMBDA_STIFF,
    "total":        1.0,         # always plot total
}

COMPONENT_STYLE = {
    "energy":       ("Energy label",  "black",          "-"),
    "energy_theta": ("Energy θ",      "dimgray",        "--"),
    "Fx":           ("Fx",            "steelblue",      "-"),
    "Fx_L4":        ("Fx L4",         "cornflowerblue", "--"),
    "Fy":           ("Fy",            "darkorange",     "-"),
    "M_left":       ("M_left",        "green",          "-"),
    "M_right":      ("M_right",       "purple",         "--"),
    "scalar":       ("Scalar total",  "gray",           ":"),
    "stiffness":    ("Stiffness",     "saddlebrown",    "--"),
    "total":        ("Total",         "brown",          "-"),
}

def active_keys(bd_list):
    """Keys that are both weighted and present in the recorded history."""
    return [
        k for k, w in COMPONENT_WEIGHTS.items()
        if w > 0 and any(k in bd for bd in bd_list)
    ]

active = active_keys(breakdown)


# ------------------------------------------------------------------
# 1) Total loss curves
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=14, fontweight="bold")

for ax, sl, title in [
    (axes[0], slice(None),       "Train vs Validation Loss (full)"),
    (axes[1], slice(mid - 1, None), f"Zoomed — Last {len(epochs) - mid + 1} Epochs"),
]:
    ax.plot(epochs[sl], train_loss[sl], label="Train", color="steelblue", linewidth=2)
    ax.plot(epochs[sl], val_loss[sl],   label="Val",   color="tomato",    linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_curves.png")


# ------------------------------------------------------------------
# 2) Validation breakdown curves (active components only)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    "Validation Loss Breakdown  (only components with non-zero weight)",
    fontsize=13, fontweight="bold",
)

for ax, sl, title in [
    (axes[0], slice(None),       "All Epochs (log scale)"),
    (axes[1], slice(mid - 1, None), f"Zoomed — Last {len(epochs) - mid + 1} Epochs"),
]:
    for key in active:
        label, color, ls = COMPONENT_STYLE[key]
        vals = [bd.get(key, 0.0) for bd in breakdown]
        ax.plot(epochs[sl], vals[sl], label=label, color=color, linestyle=ls, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_breakdown.png")


# ------------------------------------------------------------------
# 3) Test R² per output
# ------------------------------------------------------------------
auto_results = test_results.get("AUTO", {})
if not auto_results:
    raise ValueError("test_results.json does not contain 'AUTO' results.")

names      = list(auto_results.keys())
r2vals     = [auto_results[n]["R2"]     for n in names]
rmse_vals  = [auto_results[n]["RMSE"]   for n in names]
maxerr_vals= [auto_results[n]["MaxErr"] for n in names]

bar_colors = [
    "steelblue" if v >= 0.95 else "darkorange" if v >= 0.90 else "tomato"
    for v in r2vals
]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, r2vals, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.55)
ax.axhline(y=0.95, color="green", linestyle="--", linewidth=1.2, label="R² = 0.95")
ax.axhline(y=0.99, color="blue",  linestyle=":",  linewidth=1.2, label="R² = 0.99")
ax.set_ylim(min(0.0, min(r2vals) - 0.05), 1.02)
ax.set_ylabel("R² Score")
ax.set_title("Test R² per Output (AUTO)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

for bar, val in zip(bars, r2vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=9,
    )

plt.tight_layout()
plt.savefig("plots/test_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_r2.png")


# ------------------------------------------------------------------
# 4) Test RMSE / Max Error
# ------------------------------------------------------------------
x = np.arange(len(names))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w / 2, rmse_vals,   w, label="RMSE",    color="steelblue", edgecolor="black", linewidth=0.8)
b2 = ax.bar(x + w / 2, maxerr_vals, w, label="Max Err", color="tomato",    edgecolor="black", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Error (physical units)")
ax.set_title("Test RMSE and Max Error per Output (AUTO)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h * 1.02 if h != 0 else 0.01,
        f"{h:.3f}",
        ha="center", va="bottom", fontsize=8,
    )

plt.tight_layout()
plt.savefig("plots/test_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_errors.png")


# ------------------------------------------------------------------
# 5) Per-component convergence (active components only, full history)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))
for key in active:
    label, color, ls = COMPONENT_STYLE[key]
    vals = [bd.get(key, 0.0) for bd in breakdown]
    ax.plot(epochs, vals, label=label, color=color, linestyle=ls, linewidth=1.8)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss Component Convergence (active components only)")
ax.set_yscale("log")
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/gradient_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/gradient_convergence.png")
