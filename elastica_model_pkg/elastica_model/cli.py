import sys
import os
from .generation   import run_generation
from .parsing      import parse_folders
from .setup_config import show_config


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

    # ── Run AUTO generation ───────────────────────────────────
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

    print(f"\n{'='*50}")
    print(f"  uz_x range  : {uz_x_start} → {uz_x_end}")
    print(f"  Layers      : {n_layers}")
    print(f"  Succeeded   : {len(succeeded)}")
    print(f"  Failed      : {len(failed)}")
    print(f"  Blocks saved: {total}")
    print(f"  HDF5        : {os.path.abspath(hdf5_file)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
