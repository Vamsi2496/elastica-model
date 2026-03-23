import sys
import os
from .generation   import run_generation, run_generation_only_boundary
from .parsing      import parse_folders
from .setup_config import show_config
from .plotting     import plot_all_d_values, plot_3d

def main():
    print("\n=== Elastica Model — Data Generation ===\n")
    show_config()

    uz_x_start = float(input("uz_x start  (e.g. 0.60): "))
    uz_x_end   = float(input("uz_x end    (e.g. 0.99): "))
    n_layers   = int(input("Number of layers (e.g. 40): "))
    n_workers  = int(input("Number of parallel workers (default 4): ") or 4)

    hdf5_file    = input("HDF5 output file     [auto_data.h5]: ").strip() \
                   or "auto_data.h5"
    rtree_prefix = input("R-tree index prefix  [auto_rtree_index]: ").strip() \
                   or "auto_rtree_index"
    type=input("Required data, only boundary (enter 'b') or inplane also (enter 'i'): ").strip().lower()  or 'i'
    keep_folders = input("AUTO data is stored in d0p* folders. do you want to keep the folders (enter 'y') otherwise (enter 'n'): ").strip().lower() != "n"
    # ── Run AUTO generation ───────────────────────────────────
    if type == "i":
        succeeded, failed, created_folders = run_generation(
        uz_x_start, uz_x_end, n_layers,
        n_workers=n_workers
        )
    if type == "b":
        succeeded, failed, created_folders = run_generation_only_boundary(
        uz_x_start, uz_x_end, n_layers,
        n_workers=n_workers
        )
    else:
        print("Invalid input. Defaulting to 'i'")
        succeeded, failed, created_folders = run_generation(
        uz_x_start, uz_x_end, n_layers,
        n_workers=n_workers
        )
    if not created_folders:
        print("\n✗ No folders created — nothing to parse.")
        sys.exit(1)

    # ── Parse only folders created this run ──────────────────
    total = parse_folders(
        created_folders,
        hdf5_file=hdf5_file,
        rtree_prefix=rtree_prefix
    )
    # THEN delete folders if user said no
    if not keep_folders:
        import shutil
        for folder in created_folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        print(f"✓ Deleted {len(created_folders)} d0p* folders")
    print(f"\n{'='*50}")
    print(f"  uz_x range  : {uz_x_start} → {uz_x_end}")
    print(f"  Layers      : {n_layers}")
    print(f"  Succeeded   : {len(succeeded)}")
    print(f"  Failed      : {len(failed)}")
    print(f"  Blocks saved: {total}")
    print(f"  HDF5        : {os.path.abspath(hdf5_file)}")
    print(f"{'='*50}\n")
    
    # ── Plot option ───────────────────────────────────────────
    plot3d = input("Generate 3D plot? (y/n): ").strip().lower()
    if plot3d == "y":
        save_dir = input("Save plot to folder [plots_by_d]: ").strip() or "plots_by_d"
        plot_3d(
            hdf5_file=hdf5_file, save_dir=save_dir
        )
    else:
        print("Skipping 3D plot generation.")
    do_plot = input("Generate plots for all d values? (y/n): ").strip().lower()
    if do_plot == "y":
        save_dir = input("Save plots to folder [plots_by_d]: ").strip() or "plots_by_d"
        plot_all_d_values(
            hdf5_file=hdf5_file,
            rtree_prefix=rtree_prefix,
            save_dir=save_dir
        )
    else:
        print("Skipping plot generation.")

if __name__ == "__main__":
    main()
 