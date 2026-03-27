# elastica_model/loop.py

import importlib.resources
import json
import os
import shutil
import subprocess
import time

import h5py
import numpy as np
from rtree import index

from .config import PYTHON26, AUTO_DIR

RADIUS_D    = 0.01
RADIUS_PHI1 = 1.0
RADIUS_PHI2 = 1.0
NDIM = 4
NPAR = 9

TEMPLATE_FILES = ["el3.f90", "c.el3", "restart.auto", "s.initial"]

# ══════════════════════════════════════════════════════════════
#  INTERNAL HELPERS — R-tree / HDF5 lookup
# ══════════════════════════════════════════════════════════════

def _load_rtree(rtree_prefix):
    p           = index.Property()
    p.dimension = 3
    if (os.path.exists(f"{rtree_prefix}.dat") and
            os.path.exists(f"{rtree_prefix}.idx")):
        print(f"  R-tree loaded from {rtree_prefix}")
    else:
        raise FileNotFoundError(
            f"R-tree files not found: {rtree_prefix}.dat / .idx\n"
            f"Run run_generation() first to build the database."
        )
    return index.Index(rtree_prefix, properties=p)

def _copy_data_files_to(base_dir):
    """
    Copy el3.f90, c.el3, restart.auto, s.initial
    from elastica_model/data/ into base_dir if not already present.
    """
    data_path = importlib.resources.files("elastica_model") / "data"
    copied    = []
    for fname in TEMPLATE_FILES:
        dst = os.path.join(base_dir, fname)
        src = data_path / fname
        with importlib.resources.as_file(src) as src_path:
            shutil.copy2(str(src_path), dst)
        copied.append(fname)
    if copied:
        print(f"  ✓ Copied to working dir: {copied}")

def _cleanup_auto_files(base_dir):
    for prefix in ["b", "s", "d"]:
        for name in ["curr_data"]:
            fn = os.path.join(base_dir, f"{prefix}.{name}")
            if os.path.exists(fn):
                os.remove(fn)
    for ext in ["2", "3", "7", "8", "9"]:
        fn = os.path.join(base_dir, f"fort.{ext}")
        if os.path.exists(fn):
            os.remove(fn)
    # ── Template files copied from data\ ─────────────────────
    for fname in TEMPLATE_FILES:
        fp = os.path.join(base_dir, fname)
        if os.path.exists(fp):
            os.remove(fp)
    # ── loop-specific intermediates ───────────────────────────
    for fname in ["target_point.npz", "auto_status.json", "auto_log.txt","el3.exe","el3.o"]:
        fp = os.path.join(base_dir, fname)
        if os.path.exists(fp):
            os.remove(fp)
    print("✓ Intermediate files deleted")

def _find_nearest_in_hdf5(phi1, phi2, d, hdf5_file, rtree_prefix, n_hits=3):
    """
    R-tree nearest-neighbour search → read closest solution from HDF5.

    Parameters
    ----------
    phi1, phi2, d : float   target point
    hdf5_file     : str
    rtree_prefix  : str
    n_hits        : int     candidates to evaluate

    Returns
    -------
    params  : np.ndarray  shape (NPAR,)
    t_vals  : np.ndarray
    u1_vals : np.ndarray
    best_i  : int          HDF5 row index of the match
    dist    : float        Euclidean distance to match
    """
    idx  = _load_rtree(rtree_prefix)
    bbox = (d, phi1, phi2, d, phi1, phi2)
    hits = list(idx.nearest(bbox, n_hits, objects=True))
    idx.close()

    if not hits:
        raise RuntimeError(
            "R-tree returned 0 hits — database may be empty or "
            "target point is far outside the indexed region."
        )

    with h5py.File(hdf5_file, 'r') as f:
        best_dist = float('inf')
        best_i    = None
        for hit in hits:
            i    = hit.object
            dv   = float(f['d'][i])
            p1   = float(f['phi1'][i])
            p2   = float(f['phi2'][i])
            dist = np.sqrt(
                ((dv - d)    / RADIUS_D   ) ** 2 +
                ((p1 - phi1) / RADIUS_PHI1) ** 2 +
                ((p2 - phi2) / RADIUS_PHI2) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_i    = i

        params  = np.array(f['parameters'][best_i], dtype=float)
        t_vals  = np.array(f['t'][best_i],          dtype=float)
        u1_vals = np.array(f['u1'][best_i],          dtype=float)

        print(f"  Nearest: HDF5 idx={best_i}  "
              f"d={float(f['d'][best_i]):.6f}  "
              f"phi1={float(f['phi1'][best_i]):.6f}  "
              f"phi2={float(f['phi2'][best_i]):.6f}  "
              f"dist={best_dist:.6f}")

    return params, t_vals, u1_vals, best_i, best_dist


# ══════════════════════════════════════════════════════════════
#  INTERNAL HELPERS — build solution arrays
# ══════════════════════════════════════════════════════════════

def _build_solution_arrays(t_vals, u1_vals):
    """Compute u2 = du1/dt, u3 = ∫cos(u1)dt, u4 = ∫sin(u1)dt."""
    u2 = np.gradient(u1_vals, t_vals)
    u3 = np.zeros_like(u1_vals)
    u4 = np.zeros_like(u1_vals)
    if len(t_vals) > 1:
        dt     = np.diff(t_vals)
        u3[1:] = np.cumsum(
            0.5 * (np.cos(u1_vals[:-1]) + np.cos(u1_vals[1:])) * dt
        )
        u4[1:] = np.cumsum(
            0.5 * (np.sin(u1_vals[:-1]) + np.sin(u1_vals[1:])) * dt
        )
    return u2, u3, u4


# ══════════════════════════════════════════════════════════════
#  INTERNAL HELPERS — write s.initial
# ══════════════════════════════════════════════════════════════

def _fmt_sol_row(t, u1, u2, u3, u4):
    return (f" {t: .10E}   {u1: .10E}   {u2: .10E}"
            f"   {u3: .10E}   {u4: .10E}\n")


def _fmt_par_row(values):
    return "   ".join(f"{v: .10E}" for v in values) + "\n"


def _write_s_initial(params, t_vals, u1_vals, u2_vals, u3_vals, u4_vals,
                     dat_file):
    """Overwrite dat_file with the nearest-neighbour solution."""
    with open(dat_file, 'r') as f:
        lines = f.readlines()

    sol_lines = [
        _fmt_sol_row(t, u1, u2, u3, u4)
        for t, u1, u2, u3, u4 in
        zip(t_vals, u1_vals, u2_vals, u3_vals, u4_vals)
    ]
    n_sol = len(sol_lines)
    lines[1: 1 + n_sol] = sol_lines

    # Find parameter block: first line with exactly 7 floats after solution
    param_start = None
    for k in range(1 + n_sol, len(lines)):
        vals = lines[k].split()
        if len(vals) == 7:
            try:
                list(map(float, vals))
                param_start = k
                break
            except ValueError:
                pass

    if param_start is not None:
        lines[param_start]     = _fmt_par_row(params[:7])
        lines[param_start + 1] = _fmt_par_row(params[7:])
    else:
        print("  [WARN] Parameter block not found in s.initial — "
              "skipping param update")

    with open(dat_file, 'w') as f:
        f.writelines(lines)

    print(f"  s.initial written  "
          f"({n_sol} solution rows, params @ line {param_start})")


# ══════════════════════════════════════════════════════════════
#  INTERNAL HELPERS — parse AUTO output + append HDF5
# ══════════════════════════════════════════════════════════════

def _count_sign_changes(arr):
    if len(arr) < 2:
        return 0
    s = np.sign(arr)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    return int(np.sum(np.diff(s) != 0))


def _parse_auto_s_file(fname):
    """Parse AUTO s.* file → list of block dicts."""
    with open(fname, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    blocks = []
    i      = 0
    n      = len(lines)

    while i < n:
        nums = np.fromstring(lines[i], sep=' ')
        if nums.size != (1 + NDIM):
            i += 1
            continue

        t_list, u1_list, u2_list = [], [], []
        while i < n:
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size != (1 + NDIM):
                break
            t_list.append(nums[0])
            u1_list.append(nums[1])
            u2_list.append(nums[2])
            i += 1

        while i < n:                           # skip derivative block
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size != NDIM:
                break
            i += 1

        par = []
        while i < n:                           # PAR block — 7 floats
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size == 7:
                par.extend(nums.tolist())
                i += 1
                break
            i += 1
        if i < n:                              # PAR block — 2 floats
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size >= 2:
                par += [nums[0], nums[1]]
                i += 1

        if len(par) < NPAR:
            break

        d_val = par[2]
        phi1  = par[4] - par[5]
        phi2  = par[4] + par[5]
        u2_arr = np.array(u2_list, dtype=float)

        blocks.append({
            'd':   d_val, 'phi1': phi1, 'phi2': phi2,
            'par': par,
            'sc':  _count_sign_changes(u2_arr),
            't':   np.array(t_list,  dtype=float),
            'u1':  np.array(u1_list, dtype=float),
        })
    return blocks


def _append_to_hdf5(blocks, hdf5_file):
    """Append blocks to HDF5. Returns (start_idx, d, phi1, phi2)."""
    d_arr    = np.array([b['d']    for b in blocks], dtype=float)
    phi1_arr = np.array([b['phi1'] for b in blocks], dtype=float)
    phi2_arr = np.array([b['phi2'] for b in blocks], dtype=float)
    sc_arr   = np.array([b['sc']   for b in blocks], dtype=int)
    par_arr  = np.array([b['par']  for b in blocks], dtype=float)
    t_arr    = np.array([b['t']    for b in blocks], dtype=object)
    u1_arr   = np.array([b['u1']   for b in blocks], dtype=object)
    n        = len(blocks)

    if os.path.exists(hdf5_file):
        with h5py.File(hdf5_file, 'a') as f:
            s = len(f['d'])
            for key, data in [('d', d_arr), ('phi1', phi1_arr),
                               ('phi2', phi2_arr),
                               ('inflection_points', sc_arr)]:
                f[key].resize((s + n,))
                f[key][s:] = data
            f['parameters'].resize((s + n, par_arr.shape[1]))
            f['parameters'][s:] = par_arr
            f['t'].resize((s + n,));  f['t'][s:]  = t_arr
            f['u1'].resize((s + n,)); f['u1'][s:] = u1_arr
        print(f"  HDF5: +{n} blocks  (total {s + n})")
        return s, d_arr, phi1_arr, phi2_arr
    else:
        dt_vlen = h5py.vlen_dtype(np.dtype('float64'))
        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('d',    data=d_arr,   maxshape=(None,), chunks=True)
            f.create_dataset('phi1', data=phi1_arr, maxshape=(None,), chunks=True)
            f.create_dataset('phi2', data=phi2_arr, maxshape=(None,), chunks=True)
            f.create_dataset('inflection_points', data=sc_arr,
                             maxshape=(None,), chunks=True)
            f.create_dataset('parameters', data=par_arr,
                             maxshape=(None, par_arr.shape[1]), chunks=True)
            f.create_dataset('t',  shape=(n,), maxshape=(None,),
                             chunks=True, dtype=dt_vlen)
            f.create_dataset('u1', shape=(n,), maxshape=(None,),
                             chunks=True, dtype=dt_vlen)
            f['t'][:] = t_arr;  f['u1'][:] = u1_arr
        print(f"  HDF5: created with {n} blocks")
        return 0, d_arr, phi1_arr, phi2_arr


def _update_rtree_inplace(rtree_idx, start_idx, d_arr, phi1_arr, phi2_arr):
    """Insert new entries into the already-open in-memory R-tree."""
    for j, (dv, p1, p2) in enumerate(zip(d_arr, phi1_arr, phi2_arr)):
        i    = start_idx + j
        bbox = (dv, p1, p2, dv, p1, p2)
        rtree_idx.insert(i, bbox, obj=i)
    print(f"  R-tree: +{len(d_arr)} entries inserted in-memory")


def _make_bridge_code():
    return f"""
import os, sys
auto_dir = r"{AUTO_DIR}"
eq_dir   = os.environ.get("AUTO_EQ_DIR", ".")
os.chdir(auto_dir)
if auto_dir not in sys.path:
    sys.path.append(auto_dir)
import auto
os.chdir(eq_dir)
auto.auto("restart.auto")
"""


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════

def run_at_point(phi1, phi2, d,
                 hdf5_file      = "data.h5",
                 rtree_prefix   = "index",
                 base_dir       = None,
                 verbose        = True):
    """
    Given a target point (phi1, phi2, d), find the nearest solution
    in the HDF5 database, use that as initial guess, run AUTO, parse the
    output and append the new solution to HDF5 + R-tree.

    Parameters
    ----------
    phi1           : float   target left boundary angle
    phi2           : float   target right boundary angle
    d              : float   target clamp-to-clamp distance (normalised)
    hdf5_file      : str     HDF5 database path
    rtree_prefix   : str     R-tree index file prefix
    base_dir       : str     working directory (default: cwd)
    verbose        : bool    print AUTO stdout to terminal (default True)

    Returns
    -------
    convergence   : bool    True/False
    phi1_values   : list[float]
    phi2_values   : list[float]
    d_values      : list[float]
    hdf5_indices  : list[int]    new row indices written (empty if not converged)
    """
    n_hits=3
    base_dir = base_dir or os.path.abspath(os.getcwd())
    _cleanup_auto_files(base_dir)
    _copy_data_files_to(base_dir) 

    hdf5_file     = os.path.join(base_dir, hdf5_file)
    rtree_prefix  = os.path.join(base_dir, rtree_prefix)
    dat_file      = os.path.join(base_dir, "s.initial")
    sentinel_path = os.path.join(base_dir, "auto_status.json")
    npz_path      = os.path.join(base_dir, "target_point.npz")

    empty = [], [], [], []   # shorthand for failed returns

    # ── Step 1: nearest-neighbour lookup ─────────────────────
    print(f"\n  Target: phi1={phi1:.6f}  phi2={phi2:.6f}  d={d:.6f}")
    try:
        params, t_vals, u1_vals, nearest_idx, nearest_dist = \
            _find_nearest_in_hdf5(phi1, phi2, d,
                                  hdf5_file, rtree_prefix, n_hits)
    except Exception as e:
        print(f"  [ERROR] HDF5/R-tree lookup failed: {e}")
        return False, *empty

    # ── Step 2: build arrays and write s.initial ─────────────
    u2, u3, u4 = _build_solution_arrays(t_vals, u1_vals)
    try:
        _write_s_initial(params, t_vals, u1_vals, u2, u3, u4, dat_file)
    except Exception as e:
        print(f"  [ERROR] write_s_initial failed: {e}")
        return False, *empty

    # Write target_point.npz for bridge / restart.auto
    np.savez_compressed(npz_path,
                        target_point=np.array([phi1, phi2, d]))

    # ── Step 3: run AUTO bridge ───────────────────────────────
    print("  Running AUTO bridge...")
    bridge_code = _make_bridge_code()
    env         = os.environ.copy()
    env["AUTO_EQ_DIR"] = base_dir

    if verbose:
        stdout_dest, stderr_dest = None, None
    else:
        log_path    = os.path.join(base_dir, "auto_log.txt")
        log_fh      = open(log_path, "w")
        stdout_dest = log_fh
        stderr_dest = subprocess.STDOUT

    try:
        ret = subprocess.call(
            [PYTHON26, "-c", bridge_code],
            env=env, cwd=base_dir,
            stdout=stdout_dest, stderr=stderr_dest
        )
    finally:
        if not verbose:
            log_fh.close()

    if ret != 0:
        print(f"  [ERROR] Bridge exited with code {ret}")
        return False, *empty

    # ── Step 4: read sentinel ─────────────────────────────────
    try:
        with open(sentinel_path) as f:
            converged = json.load(f)["converged"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"  [ERROR] Cannot read sentinel: {e}")
        return False, *empty

    if not converged:
        print("  AUTO did NOT converge.")
        return False, *empty

    # ── Step 5: parse output → HDF5 + R-tree ─────────────────
    print("  AUTO converged — parsing output...")
    s_curr = os.path.join(base_dir, "s.curr_data")
    blocks = _parse_auto_s_file(s_curr)

    if not blocks:
        print("  [WARN] No blocks parsed from s.curr_data")
        return True, *empty

    start_idx, d_arr, phi1_arr, phi2_arr = _append_to_hdf5(blocks, hdf5_file)

    # Update the persistent on-disk R-tree
    p   = index.Property(); p.dimension = 3
    idx = index.Index(rtree_prefix, properties=p)
    _update_rtree_inplace(idx, start_idx, d_arr, phi1_arr, phi2_arr)
    idx.close()

    hdf5_indices = list(range(start_idx, start_idx + len(blocks)))
    d_values     = d_arr.tolist()
    phi1_values  = phi1_arr.tolist()
    phi2_values  = phi2_arr.tolist()

    print(f"  Done  → {len(blocks)} new blocks  "
          f"(HDF5 idx {start_idx} → {start_idx + len(blocks) - 1})")
    _cleanup_auto_files(base_dir)
    return True, phi1_values, phi2_values, d_values, hdf5_indices