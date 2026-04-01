# diagnose.py
import h5py
import numpy as np

with h5py.File("data.h5", "r") as f:
    print("=== ALL KEYS ===")
    for key in f.keys():
        ds = f[key]
        print(f"  {key:<20} shape={ds.shape}  dtype={ds.dtype}")

    print("\n=== SAMPLE [0] ===")
    for key in f.keys():
        val = f[key][0]
        print(f"  {key:<20} type={type(val)}  shape={np.array(val).shape}  val={np.array(val)[:5] if np.array(val).ndim > 0 else val}")