import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from rtree import index as rtree_index

def plot_3d(hdf5_file="auto_data.h5",save_dir=None):
    """3D plot with d, phi1, phi2 as axes. Colored by PAR(9), marker by inflection points."""



    # Load all data
    initial=0
    with h5py.File(hdf5_file, 'r') as f:
        d = f['d'][initial:]
        phi1 = f['phi1'][initial:]
        phi2 = f['phi2'][initial:]
        par = f['parameters'][initial:]
        inflection_points = f['inflection_points'][initial:]
    
    par9 = par[:, 8]


    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define markers
    #markers = {0: 's', 1: '^', 2: 'o'}
    markers = {0: 's', 1: '^', 2: 'o', 3: 'D', 'other': 'x'}
    
    # Plot each inflection point group
    for sign_change_value in [0, 1, 2, 3]:
        mask = (inflection_points == sign_change_value)
        if np.any(mask):
            scatter = ax.scatter(phi1[mask], phi2[mask], d[mask],
                                c=par9[mask],
                                marker=markers[sign_change_value],
                                cmap='jet',
                                s=10,
                                label=f'Inflection points={sign_change_value}',
                                alpha=0.6,
                                edgecolors='black',
                                linewidths=0.3,
                                vmin=par9.min(),
                                vmax=par9.max())
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Energy', fontsize=12)
    
    # Labels
    
    ax.set_xlabel('phi1', fontsize=12)
    ax.set_ylabel('phi2', fontsize=12)
    ax.set_zlabel('d', fontsize=12)
    ax.set_title('3D Bifurcation Diagram', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    
    #ax.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    # ── Save ─────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir or "", "3D_plot.png")
    fig.savefig(fname, dpi=150)
    print(f"✓ 3D plot saved → {os.path.abspath(fname)}")
    plt.close(fig)



def plot_bifurcation_at_d(d_target, hdf5_file="auto_data.h5",
                           rtree_prefix="auto_rtree_index",
                           tolerance=1e-6, save_dir=None):
    """
    Plot phi1 vs phi2 at a specific d value.
    Colored by PAR(9), marker by inflection points.
    """
    p = rtree_index.Property()
    p.dimension = 3
    idx = rtree_index.Index(rtree_prefix, properties=p)

    try:
        query_box = (
            d_target - tolerance, -1000, -1000,
            d_target + tolerance,  1000,  1000
        )
        matches       = list(idx.intersection(query_box, objects=True))
        match_indices = sorted([hit.object for hit in matches])
    finally:
        idx.close()

    print(f"  Found {len(match_indices)} points at d = {d_target:.6f}")
    if not match_indices:
        return

    with h5py.File(hdf5_file, "r") as f:
        phi1              = f["phi1"][match_indices]
        phi2              = f["phi2"][match_indices]
        par               = f["parameters"][match_indices]
        inflection_points = f["inflection_points"][match_indices]

    par9 = par[:, 8]

    markers      = {0: "s", 1: "^", 2: "o", 3: "D", "other": "x"}
    marker_sizes = {0: 20,  1: 20,  2: 5,   3: 20,  "other": 20}

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = None
    vmin, vmax = par9.min(), par9.max()

    for sc_val in [0, 1, 2, 3]:
        mask = (inflection_points == sc_val)
        if np.any(mask):
            scatter = ax.scatter(
                phi1[mask], phi2[mask],
                c=par9[mask], marker=markers[sc_val],
                cmap="jet", s=marker_sizes[sc_val],
                label=f"Inflection pts = {sc_val}",
                alpha=0.7, edgecolors="black",
                linewidths=0.5, vmin=vmin, vmax=vmax
            )

    other_mask = (inflection_points > 3) | (inflection_points < 0)
    if np.any(other_mask):
        scatter = ax.scatter(
            phi1[other_mask], phi2[other_mask],
            c=par9[other_mask], marker=markers["other"],
            cmap="jet", s=marker_sizes["other"],
            label="Inflection pts = other",
            alpha=0.7, linewidths=0.5,
            vmin=vmin, vmax=vmax
        )

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Energy (PAR9)", fontsize=12)

    ax.set_xlabel("phi1", fontsize=12)
    ax.set_ylabel("phi2", fontsize=12)
    ax.set_title(f"d = {d_target:.6f})", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"phi1_vs_phi2_d_{d_target:.6f}.png"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
    fig.savefig(fname, dpi=150)
    print(f"  Saved → {fname}")
    plt.close(fig)


def plot_all_d_values(hdf5_file="auto_data.h5",
                      rtree_prefix="auto_rtree_index",
                      tolerance=1e-6, save_dir="plots_by_d"):
    """Plot phi1 vs phi2 for every unique d value found in HDF5."""
    with h5py.File(hdf5_file, "r") as f:
        d_all = f["d"][:]

    unique_d = np.unique(np.round(d_all, 6))
    print(f"\nFound {len(unique_d)} unique d values → generating {len(unique_d)} plots\n")

    for i, dval in enumerate(unique_d, 1):
        print(f"[{i}/{len(unique_d)}] d = {dval:.4f}")
        plot_bifurcation_at_d(
            dval,
            hdf5_file=hdf5_file,
            rtree_prefix=rtree_prefix,
            tolerance=tolerance,
            save_dir=save_dir
        )

    print(f"\n✓ All {len(unique_d)} plots saved to '{save_dir}'")


if __name__ == "__main__":
    # ── Terminal prompt when run directly ─────────────────────
    print("\n=== Elastica Model — Plotting ===\n")

    hdf5_file    = input("HDF5 file         [auto_data.h5]: ").strip()      or "auto_data.h5"
    rtree_prefix = input("R-tree prefix     [auto_rtree_index]: ").strip()  or "auto_rtree_index"
    save_dir     = input("Save plots to     [plots_by_d]: ").strip()        or "plots_by_d"
    tolerance    = float(input("d tolerance       [0.000001]: ").strip() or 1e-6)

    plot_all_d_values(
        hdf5_file=hdf5_file,
        rtree_prefix=rtree_prefix,
        tolerance=tolerance,
        save_dir=save_dir
    )
