import json
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# ── Load files ────────────────────────────────────────────────────── #
with open("training_history.json", "r") as f:
    history = json.load(f)

with open("test_results.json", "r") as f:
    test_results = json.load(f)

train_loss = history["train"]
val_loss   = history["val"]
breakdown  = history["breakdown"]   # list of dicts, one per epoch
epochs     = list(range(1, len(train_loss) + 1))
mid        = len(epochs) // 2       # halfway point for zoom plot


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 1 — Train vs Validation Loss (full + zoomed)
# ═══════════════════════════════════════════════════════════════════ #
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=14, fontweight="bold")

# Left: full log-scale loss curve
ax = axes[0]
ax.plot(epochs, train_loss, label="Train", color="steelblue", linewidth=2)
ax.plot(epochs, val_loss,   label="Val",   color="tomato",
        linewidth=2, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss")
ax.set_title("Train vs Validation Loss")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: zoomed into last 50% of epochs
ax2 = axes[1]
ax2.plot(epochs[mid:], train_loss[mid:], label="Train",
         color="steelblue", linewidth=2)
ax2.plot(epochs[mid:], val_loss[mid:],   label="Val",
         color="tomato", linewidth=2, linestyle="--")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Total Loss")
ax2.set_title(f"Zoomed — Last {len(epochs) - mid} Epochs")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/loss_curves.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 2 — Loss Breakdown per Epoch
# Keys: scalar, BC_moment, equilibrium, consistency
# ═══════════════════════════════════════════════════════════════════ #
breakdown_keys = {
    "scalar"     : ("Scalar MSE (Fx,Fy,ML,MR)", "steelblue"),
    "BC_moment"  : ("BC Moment (M at s=0,L)",    "green"),
    "equilibrium": ("Equilibrium ODE",            "purple"),
    "consistency": ("Consistency (θ→F check)",    "crimson"),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Validation Loss Breakdown", fontsize=14, fontweight="bold")

# Left: all components on log scale
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

# Right: zoomed last 50%, linear scale — shows fine convergence
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


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 3 — Test R² Bar Chart
# ═══════════════════════════════════════════════════════════════════ #
names  = list(test_results.keys())          # Fx, Fy, M_left, M_right
r2vals = [test_results[n]["R2"] for n in names]

bar_colors = ["steelblue" if v >= 0.95 else
              "darkorange" if v >= 0.90 else
              "tomato" for v in r2vals]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, r2vals, color=bar_colors, edgecolor="black",
              linewidth=0.8, width=0.5)
ax.axhline(y=0.95, color="green", linestyle="--", linewidth=1.2,
           label="R²=0.95 target")
ax.axhline(y=0.99, color="blue",  linestyle=":",  linewidth=1.2,
           label="R²=0.99 target")
ax.set_ylim(max(0.0, min(r2vals) - 0.05), 1.02)
ax.set_ylabel("R² Score")
ax.set_title("Test R² per Output")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("plots/test_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/test_r2.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 4 — RMSE and MaxErr per scalar output
# ═══════════════════════════════════════════════════════════════════ #
scalar_names = list(test_results.keys())
rmse_vals    = [test_results[n]["RMSE"]    for n in scalar_names]
maxerr_vals  = [test_results[n]["MaxErr"]  for n in scalar_names]

x = np.arange(len(scalar_names))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, rmse_vals,   w, label="RMSE",    color="steelblue",
            edgecolor="black", linewidth=0.8)
b2 = ax.bar(x + w/2, maxerr_vals, w, label="Max Err", color="tomato",
            edgecolor="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(scalar_names)
ax.set_ylabel("Error (physical units)")
ax.set_title("Test RMSE and Max Error per Scalar Output")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

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
# FIGURE 5 — Physics Loss Convergence (BC + Equilibrium separately)
# Shows how well the predicted scalars satisfy the elastica ODE
# ═══════════════════════════════════════════════════════════════════ #
bc_vals  = [bd.get("BC_moment",   0.0) for bd in breakdown]
eq_vals  = [bd.get("equilibrium", 0.0) for bd in breakdown]
con_vals = [bd.get("consistency", 0.0) for bd in breakdown]

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(epochs, bc_vals,  label="BC Moment Loss",    color="green",
        linewidth=1.8)
ax.plot(epochs, eq_vals,  label="Equilibrium ODE",   color="purple",
        linewidth=1.8)
ax.plot(epochs, con_vals, label="Consistency Loss",  color="crimson",
        linewidth=1.8, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Physics Constraint Convergence")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate final values
for vals, color, label in [
    (bc_vals,  "green",  "BC"),
    (eq_vals,  "purple", "Eq"),
    (con_vals, "crimson","Con"),
]:
    ax.annotate(f"{label}={vals[-1]:.4f}",
                xy=(epochs[-1], vals[-1]),
                xytext=(-45, 10),
                textcoords="offset points",
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

plt.tight_layout()
plt.savefig("plots/physics_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/physics_convergence.png")


# ═══════════════════════════════════════════════════════════════════ #
# FIGURE 6 — Final Epoch Summary Table (text figure)
# ═══════════════════════════════════════════════════════════════════ #
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table_data = [["Output", "R²", "RMSE", "Max Error"]]
for name in scalar_names:
    r = test_results[name]
    table_data.append([
        name,
        f"{r['R2']:.5f}",
        f"{r['RMSE']:.4e}",
        f"{r['MaxErr']:.4e}",
    ])

table = ax.table(cellText  = table_data[1:],
                 colLabels = table_data[0],
                 loc       = "center",
                 cellLoc   = "center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 1.8)

# Header styling
for j in range(4):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Row colouring by R² value
for i, name in enumerate(scalar_names):
    r2v = test_results[name]["R2"]
    color = "#d5f5e3" if r2v >= 0.95 else \
            "#fdebd0" if r2v >= 0.90 else "#fadbd8"
    for j in range(4):
        table[i + 1, j].set_facecolor(color)

ax.set_title("Test Results Summary", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("plots/results_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → plots/results_table.png")


# ── Summary ───────────────────────────────────────────────────────── #
print("\nAll plots saved in  plots/")
print("  plots/loss_curves.png")
print("  plots/loss_breakdown.png")
print("  plots/test_r2.png")
print("  plots/test_errors.png")
print("  plots/physics_convergence.png")
print("  plots/results_table.png")