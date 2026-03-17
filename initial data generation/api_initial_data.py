# api_initial_data.py
import sys
import numpy as np
import os
import subprocess

PYTHON26 = r"C:\Python26\python.exe"
BASE_DIR = os.path.abspath(os.getcwd())
HDF5_FILE    = "auto_data.h5"
RTREE_PREFIX = "auto_rtree_index"

# ── 1. Terminal inputs ────────────────────────────────────────
print("\n=== Elastica Initial Data Generation ===\n")
uz_x_start = float(input("uz_x start (e.g. 0.60): "))
uz_x_end   = float(input("uz_x end   (e.g. 0.99): "))
uz_x_step  = float(input("uz_x step  (e.g. 0.01): "))

uz_x_list = [round(v, 4) for v in
             np.arange(uz_x_start, uz_x_end + uz_x_step / 2, uz_x_step)]
print(f"\n{len(uz_x_list)} iterations: {uz_x_list}\n")

# ── 2. Save uz_x_list to .npz ────────────────────────────────
npz_path = os.path.join(BASE_DIR, "uz_x_list.npz")
np.savez(npz_path, uz_x_list=np.array(uz_x_list))
print(f"✓ uz_x_list saved to {npz_path}\n")

# ── 3. Run AUTO via Python 2.6 inline ────────────────────────
bridge_code = """
import os, sys
auto_dir = "C:\\\\MinGW\\\\msys\\\\1.0\\\\home\\\\sanch\\\\auto\\\\07p\\\\python"
eq_dir   = os.environ.get("AUTO_EQ_DIR", ".")
os.chdir(auto_dir)
if auto_dir not in sys.path:
    sys.path.append(auto_dir)
import auto
os.chdir(eq_dir)
auto.auto("initial_data_generation.auto")
"""

env = os.environ.copy()
env["AUTO_EQ_DIR"] = BASE_DIR

print("Running AUTO (Python 2.6)...\n")
ret = subprocess.call([PYTHON26, "-c", bridge_code], env=env, cwd=BASE_DIR)

if ret != 0:
    print(f"\n✗ AUTO failed with exit code {ret}")
    sys.exit(ret)

print("\n✓ AUTO run complete\n")

# ── 4. Run automated_parsing.py ──────────────────────────────
print("Running automated_parsing.py...\n")
subprocess.run([sys.executable, "automated_parsing.py"], check=True)
print("\n✓ Parsing complete")
print(f"✓ Data saved to {HDF5_FILE}")
