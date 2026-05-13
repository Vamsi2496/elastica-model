import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import Config

os.makedirs("plots", exist_ok=True)

with open("training_history.json", "r") as f:
    history = json.load(f)

with open("test_results.json", "r") as f:
    test_results = json.load(f)

train_loss = history.get("train", [])
val_loss   = history.get("val",   [])
breakdown  = history.get("breakdown", [])
epochs     = list(range(1, len(train_loss) + 1))

if len(epochs) == 0:
    raise ValueError("training_history.json contains no epochs.")

# ------------------------------------------------------------------
# Breakdown components present in this 2-input model
# (Energy label + M_left + M_right via gradients — no Fx, no Fy)
# ------------------------------------------------------------------
W = Config.W_SCALAR

COMPONENT_WEIGHTS = {
    "energy":  Config.W_ENERGY_LABEL,
    "M_left":  W * Config.M_WEIGHT,
    "M_right": W * Config.M_WEIGHT,
    "scalar":  W,
    "total":   1.0,
}

COMPONENT_STYLE = {
    "energy":  ("Energy",       "black",   "-"),
    "M_left":  ("M_left",       "green",   "-"),
    "M_right": ("M_right",      "purple",  "--"),
    "scalar":  ("Scalar total", "gray",    ":"),
    "total":   ("Total",        "brown",   "-"),
}

ATTR_TO_KEY = {
    "W_ENERGY_LABEL": "energy",
    "M_WEIGHT":       "M_left",
}


def active_keys(bd_list):
    return [
        k for k, w in COMPONENT_WEIGHTS.items()
        if w > 0 and any(k in bd for bd in bd_list)
    ]


active = active_keys(breakdown)


def add_schedule_markers(ax, x_min, x_max):
    for attr, intro, ramp, _init in Config.LOSS_SCHEDULE:
        if intro < x_min or intro > x_max:
            continue
        key   = ATTR_TO_KEY.get(attr)
        color = COMPONENT_STYLE[key][1] if key and key in COMPONENT_STYLE else "gray"
        ax.axvline(intro, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
        if ramp > 0:
            ax.axvspan(intro, min(intro + ramp, x_max), color=color, alpha=0.08)
        ax.text(intro, 0.97, f" {attr} on",
                color=color, fontsize=8, rotation=90,
                va="top", ha="left", alpha=0.85,
                transform=ax.get_xaxis_transform())


# ------------------------------------------------------------------
# 1) Total loss curves
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Training History", fontsize=14, fontweight="bold")
ax.plot(epochs, train_loss, label="Train", color="steelblue", linewidth=2)
ax.plot(epochs, val_loss,   label="Val",   color="tomato",    linewidth=2, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss")
ax.set_title("Train vs Validation Loss")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_curves.png")

# ------------------------------------------------------------------
# 2) Validation breakdown
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Validation Loss Breakdown", fontsize=13, fontweight="bold")
for key in active:
    label, color, ls = COMPONENT_STYLE[key]
    vals = [bd.get(key, 0.0) for bd in breakdown]
    ax.plot(epochs, vals, label=label, color=color, linestyle=ls, linewidth=1.8)
add_schedule_markers(ax, 1, len(epochs))
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Value")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_breakdown.png")

# ------------------------------------------------------------------
# 3) Test R² per output
# ------------------------------------------------------------------
names       = list(test_results.keys())
r2vals      = [test_results[n]["R2"]     for n in names]
rmse_vals   = [test_results[n]["RMSE"]   for n in names]
maxerr_vals = [test_results[n]["MaxErr"] for n in names]

bar_colors = [
    "steelblue" if v >= 0.95 else "darkorange" if v >= 0.90 else "tomato"
    for v in r2vals
]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(names, r2vals, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.45)
ax.axhline(y=0.95, color="green", linestyle="--", linewidth=1.2, label="R²=0.95")
ax.axhline(y=0.99, color="blue",  linestyle=":",  linewidth=1.2, label="R²=0.99")
ax.set_ylim(min(0.0, min(r2vals) - 0.05), 1.02)
ax.set_ylabel("R² Score")
ax.set_title("Test R² per Output")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/test_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_r2.png")

# ------------------------------------------------------------------
# 4) RMSE / Max Error
# ------------------------------------------------------------------
x = np.arange(len(names))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, rmse_vals,   w, label="RMSE",    color="steelblue", edgecolor="black", linewidth=0.8)
b2 = ax.bar(x + w/2, maxerr_vals, w, label="Max Err", color="tomato",    edgecolor="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Error (physical units)")
ax.set_title("Test RMSE and Max Error per Output")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02 if h != 0 else 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plots/test_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_errors.png")

# ------------------------------------------------------------------
# 5) Weighted component convergence
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))
for key in active:
    label, color, ls = COMPONENT_STYLE[key]
    weight = COMPONENT_WEIGHTS[key]
    raw_vals      = [bd.get(key, 0.0) for bd in breakdown]
    weighted_vals = [v * weight for v in raw_vals]
    ax.plot(epochs, weighted_vals,
            label=f"{label} (×{weight:.3g})",
            color=color, linestyle=ls, linewidth=1.8)
add_schedule_markers(ax, 1, len(epochs))
ax.set_xlabel("Epoch")
ax.set_ylabel("Weighted Loss")
ax.set_title("Loss Component Convergence (weighted)")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/gradient_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/gradient_convergence.png")
