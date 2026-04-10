# master_loop.py  (Python 3)
# Replaces both the batch file AND the two spawn scripts.
# Only bridge.py is kept as a subprocess (needs Python 2.6 for AUTO).

import subprocess, json, os, sys, time
import numpy as np
import trimesh
import h5py
from rtree import index

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these
# ═══════════════════════════════════════════════════════════════════════════════
N_POINTS    = 35000
EQ_DIR      = r"C:\Users\sanch\Desktop\Restart solution"
MESH_FILE   = "surface mesh poisson better.ply"
HDF5_FILE   = "auto_data.h5"
RTREE_BASE  = "auto_rtree_index"
DAT_FILE    = "s.initial"
LOG_FILE    = "pipeline_log.txt"
BRIDGE_EXE  = r"C:\Python26\python.exe"
BRIDGE_SCRIPT = "bridge.py"

D_SCALE     = 400       # raw d / D_SCALE gives normalised d
BATCH_SIZE  = 5        # points tested per sampling iteration
NDIM        = 4
NPAR        = 9
RADIUS_D    = 0.01
RADIUS_PHI1 = 1.0
RADIUS_PHI2 = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  ONE-TIME LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_mesh(mesh_file):
    print("Loading surface mesh...")
    raw  = trimesh.load(mesh_file, force='mesh')
    mesh = trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces)
    # Uncomment if pyembree is installed (5-10x faster contains()):
    mesh.use_embree = True
    print(f"  Watertight: {mesh.is_watertight},  Volume: {mesh.volume:.4f}")
    return mesh


def load_rtree(base):
    p = index.Property()
    p.dimension = 3
    if os.path.exists(f"{base}.dat") and os.path.exists(f"{base}.idx"):
        print("Loading R-tree index from disk...")
    else:
        print("R-tree index not found — creating empty index.")
    return index.Index(base, properties=p)   # stays open all iterations


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 HELPERS  — sample a point, look up nearest solution, write s.initial
# ═══════════════════════════════════════════════════════════════════════════════

def sample_point_inside_mesh(mesh):
    """Randomly sample batches until a point inside the mesh is found."""
    verts     = mesh.vertices
    phi1_min, phi1_max = verts[:, 0].min(), verts[:, 0].max()
    phi2_min, phi2_max = verts[:, 1].min(), verts[:, 1].max()
    d_min, d_max       = 0.60 * D_SCALE, 0.99 * D_SCALE

    lo = np.array([phi1_min, phi2_min, d_min])
    hi = np.array([phi1_max, phi2_max, d_max])

    n_tested = iteration = 0
    found    = None
    while found is None:
        iteration += 1
        batch     = np.random.uniform(lo, hi, size=(BATCH_SIZE, 3))
        mask      = mesh.contains(batch)
        n_tested += BATCH_SIZE
        if mask.any():
            found = batch[np.argmax(mask)]

    print(f"  Sampled point after {n_tested} tries ({iteration} iter(s))")
    print(f"  phi1={found[0]:.6f}  phi2={found[1]:.6f}  d={found[2]/D_SCALE:.6f}")
    return found


def find_closest_in_hdf5(found_point, idx):
    """R-tree neighbourhood search → HDF5 read of closest stored solution."""
    s_phi1 = found_point[0]
    s_phi2 = found_point[1]
    s_d    = found_point[2] / D_SCALE

    # FIX: lower-d bound was `s_d` (not `s_d - RADIUS_D`) in original code
    query_bbox = (
        s_d    ,    s_phi1 - RADIUS_PHI1,  s_phi2 - RADIUS_PHI2,
        s_d    + RADIUS_D,    s_phi1 + RADIUS_PHI1,  s_phi2 + RADIUS_PHI2,
    )

    hits = list(idx.intersection(query_bbox, objects=True))
    print(f"  R-tree hits: {len(hits)}")
    if not hits:
        print("  No hits — increase RADIUS_* or check index.")
        return None

    with h5py.File(HDF5_FILE, 'r') as f:
        best_dist   = float('inf')
        best_idx    = None
        for hit in hits:
            i  = hit.object
            dv = f['d'][i];  p1 = f['phi1'][i];  p2 = f['phi2'][i]
            dist = np.sqrt(
                ((dv - s_d)    / RADIUS_D   ) ** 2 +
                ((p1 - s_phi1) / RADIUS_PHI1) ** 2 +
                ((p2 - s_phi2) / RADIUS_PHI2) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        params  = f['parameters'][best_idx]
        t_vals  = f['t'][best_idx]
        u1_vals = f['u1'][best_idx]
        print(f"  Closest: HDF5 idx={best_idx}  "
              f"d={f['d'][best_idx]:.6f}  "
              f"phi1={f['phi1'][best_idx]:.6f}  "
              f"phi2={f['phi2'][best_idx]:.6f}  "
              f"dist={best_dist:.4f}")

    return params, t_vals, u1_vals


def build_solution_arrays(t_vals, u1_vals):
    """Compute u2 = du1/dt, u3 = ∫cos(u1)dt, u4 = ∫sin(u1)dt."""
    u2 = np.gradient(u1_vals, t_vals)
    u3 = np.zeros_like(u1_vals)
    u4 = np.zeros_like(u1_vals)
    if len(t_vals) > 1:
        dt = np.diff(t_vals)
        u3[1:] = np.cumsum(0.5 * (np.cos(u1_vals[:-1]) + np.cos(u1_vals[1:])) * dt)
        u4[1:] = np.cumsum(0.5 * (np.sin(u1_vals[:-1]) + np.sin(u1_vals[1:])) * dt)
    return u2, u3, u4


def _fmt_sol_row(t, u1, u2, u3, u4):
    return (f" {t: .10E}   {u1: .10E}   {u2: .10E}"
            f"   {u3: .10E}   {u4: .10E}\n")


def _fmt_par_row(values):
    return "   ".join(f"{v: .10E}" for v in values) + "\n"


def write_s_initial(params, t_vals, u1_vals, u2_vals, u3_vals, u4_vals):
    """Overwrite s.initial with the nearest-neighbour solution."""
    with open(DAT_FILE, 'r') as f:
        lines = f.readlines()

    sol_lines = [
        _fmt_sol_row(t, u1, u2, u3, u4)
        for t, u1, u2, u3, u4 in zip(t_vals, u1_vals, u2_vals, u3_vals, u4_vals)
    ]
    n_sol = len(sol_lines)
    lines[1 : 1 + n_sol] = sol_lines   # rows 2…(n_sol+1) in 1-indexed

    # Robust parameter-block finder: first 7-float line after the solution block
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
        print("  [WARN] Parameter block not found in s.initial — skipping param update")

    with open(DAT_FILE, 'w') as f:
        f.writelines(lines)
    print(f"  s.initial written ({n_sol} rows, params @ line {param_start})")


def sample_and_write(mesh, idx):
    """
    Full Step 1:
      sample random inside-point → R-tree / HDF5 lookup → write s.initial
    Returns found_point (3-element array) or None on failure.
    """
    found_point = sample_point_inside_mesh(mesh)

    result = find_closest_in_hdf5(found_point, idx)
    if result is None:
        return None

    params, t_vals, u1_vals = result
    u2, u3, u4 = build_solution_arrays(t_vals, u1_vals)
    write_s_initial(params, t_vals, u1_vals, u2, u3, u4)

    # bridge.py / restart.auto read this file
    np.savez_compressed('target_point.npz', target_point=found_point)
    return found_point


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 HELPERS  — parse AUTO output, append HDF5, update in-memory R-tree
# ═══════════════════════════════════════════════════════════════════════════════

def _read_stripped(fname):
    with open(fname, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]


def _count_sign_changes(arr):
    if len(arr) < 2:
        return 0
    s = np.sign(arr)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    return int(np.sum(np.diff(s) != 0))


def parse_auto_s_file(fname, phi_threshold=None):
    """
    Parse AUTO s.* file.
    Mesh block  : (1+NDIM) floats/line → [t u1 u2 u3 u4]
    Deriv block : NDIM floats/line (skipped)
    PAR block   : 7 floats on one line, then 2 floats on the next
    """
    lines  = _read_stripped(fname)
    blocks = []
    i = nlines = 0
    nlines = len(lines)

    while i < nlines:
        nums = np.fromstring(lines[i], sep=' ')
        if nums.size != (1 + NDIM):
            i += 1
            continue

        t_list, u1_list, u2_list = [], [], []
        while i < nlines:
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size != (1 + NDIM):
                break
            t_list.append(nums[0]); u1_list.append(nums[1]); u2_list.append(nums[2])
            i += 1

        # skip derivative block
        while i < nlines:
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size != NDIM:
                break
            i += 1

        # PAR block: 7 floats then 2 floats
        par = []
        while i < nlines:
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size == 7:
                par.extend(nums.tolist())
                i += 1
                break
            i += 1
        if i < nlines:
            nums = np.fromstring(lines[i], sep=' ')
            if nums.size >= 2:
                par += [nums[0], nums[1]]
                i += 1

        if len(par) < NPAR:
            break

        d    = par[2]
        phi1 = par[4] - par[5]
        phi2 = par[4] + par[5]
        u2_arr = np.array(u2_list, dtype=float)

        if phi_threshold is None or (abs(phi1) <= phi_threshold and
                                     abs(phi2) <= phi_threshold):
            blocks.append({
                'd': d, 'phi1': phi1, 'phi2': phi2,
                'par': par,
                'sc': _count_sign_changes(u2_arr),
                't':  np.array(t_list,  dtype=float),
                'u1': np.array(u1_list, dtype=float),
            })
    return blocks


def append_to_hdf5(blocks):
    """Append parsed blocks to HDF5. Returns (start_idx, d, phi1, phi2)."""
    if not blocks:
        return None

    d    = np.array([b['d']    for b in blocks], dtype=float)
    phi1 = np.array([b['phi1'] for b in blocks], dtype=float)
    phi2 = np.array([b['phi2'] for b in blocks], dtype=float)
    sc   = np.array([b['sc']   for b in blocks], dtype=int)
    par  = np.array([b['par']  for b in blocks], dtype=float)
    t_a  = np.array([b['t']    for b in blocks], dtype=object)
    u1_a = np.array([b['u1']   for b in blocks], dtype=object)
    n    = len(d)

    if os.path.exists(HDF5_FILE):
        with h5py.File(HDF5_FILE, 'a') as f:
            s = len(f['d'])
            for key, data in [('d', d), ('phi1', phi1), ('phi2', phi2),
                               ('inflection_points', sc)]:
                f[key].resize((s + n,));  f[key][s:] = data
            f['parameters'].resize((s + n, par.shape[1]))
            f['parameters'][s:] = par
            f['t'].resize((s + n,));   f['t'][s:]  = t_a
            f['u1'].resize((s + n,));  f['u1'][s:] = u1_a
        print(f"  HDF5: +{n} points (total {s + n})")
        return s, d, phi1, phi2
    else:
        dt_vlen = h5py.vlen_dtype(np.dtype('float64'))
        with h5py.File(HDF5_FILE, 'w') as f:
            f.create_dataset('d',                data=d,   maxshape=(None,), chunks=True)
            f.create_dataset('phi1',             data=phi1, maxshape=(None,), chunks=True)
            f.create_dataset('phi2',             data=phi2, maxshape=(None,), chunks=True)
            f.create_dataset('inflection_points',data=sc,  maxshape=(None,), chunks=True)
            f.create_dataset('parameters', data=par,
                             maxshape=(None, par.shape[1]), chunks=True)
            f.create_dataset('t',  shape=(n,), maxshape=(None,), chunks=True, dtype=dt_vlen)
            f.create_dataset('u1', shape=(n,), maxshape=(None,), chunks=True, dtype=dt_vlen)
            f['t'][:] = t_a;  f['u1'][:] = u1_a
        print(f"  HDF5: created with {n} points")
        return 0, d, phi1, phi2


def update_rtree_inplace(idx, start_idx, d, phi1, phi2):
    """Insert new entries into the already-open in-memory R-tree (no disk re-open)."""
    for i, (dv, p1, p2) in enumerate(zip(d, phi1, phi2), start=start_idx):
        bbox = (dv, p1, p2, dv, p1, p2)
        idx.insert(i, bbox, obj=i)
    print(f"  R-tree: +{len(d)} entries")


def parse_and_append(idx):
    """Full Step 3: parse s.curr_data → HDF5 → in-memory R-tree."""
    blocks = parse_auto_s_file("s.curr_data")
    if not blocks:
        print("  [WARN] No blocks parsed from s.curr_data")
        return
    result = append_to_hdf5(blocks)
    if result is None:
        return
    start_idx, d, phi1, phi2 = result
    update_rtree_inplace(idx, start_idx, d, phi1, phi2)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log(label, msg):
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{label}] {time.strftime('%H:%M:%S')}  {msg}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_pipeline = time.time()

    # ── load heavy objects once ──────────────────────────────────────────────
    mesh = load_mesh(MESH_FILE)
    idx  = load_rtree(RTREE_BASE)

    converged_count = failed_count = 0
    sentinel_path   = os.path.join(EQ_DIR, "auto_status.json")

    with open(LOG_FILE, 'w') as log:
        log.write(f"Pipeline started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("=" * 50 + "\n")

    for total in range(1, N_POINTS + 1):
        t_iter = time.time()
        print(f"\n[Iter {total}/{N_POINTS}] " + "-" * 38)

        # ── Step 1: sample + write s.initial ────────────────────────────────
        try:
            point = sample_and_write(mesh, idx)
        except Exception as e:
            print(f"  [SKIP] sample_and_write error: {e}")
            failed_count += 1
            _log(total, f"SKIPPED - sample_and_write: {e}")
            continue

        if point is None:
            print("  [SKIP] No R-tree hit in region — increase RADIUS_* values")
            failed_count += 1
            _log(total, "SKIPPED - no R-tree hit")
            continue

        # ── Step 2: run AUTO via Python 2.6 bridge ──────────────────────────
        print("  Running bridge.py (AUTO)...")
        ret = subprocess.call([BRIDGE_EXE, BRIDGE_SCRIPT])
        if ret != 0:
            print(f"  [SKIP] bridge.py exited with code {ret}")
            failed_count += 1
            _log(total, f"SKIPPED - bridge.py exit {ret}")
            continue

        # ── Step 3: read sentinel ────────────────────────────────────────────
        try:
            with open(sentinel_path) as f:
                converged = json.load(f)["converged"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"  [SKIP] Cannot read sentinel: {e}")
            failed_count += 1
            _log(total, f"SKIPPED - sentinel error: {e}")
            continue

        # ── Step 4: parse or skip ────────────────────────────────────────────
        if converged:
            print("  AUTO converged — parsing output...")
            try:
                parse_and_append(idx)
                converged_count += 1
                _log(total, "SUCCESS")
            except Exception as e:
                print(f"  [WARN] Parser error: {e}")
                _log(total, f"CONVERGED but parser failed: {e}")
                converged_count += 1   # AUTO succeeded; just log the parser issue
        else:
            print("  AUTO did NOT converge.")
            failed_count += 1
            _log(total, "NOT CONVERGED")

        print(f"  Iter time: {time.time() - t_iter:.2f}s")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_pipeline
    print("\n" + "=" * 50)
    print(f"Pipeline complete.")
    print(f"  Total     : {N_POINTS}")
    print(f"  Converged : {converged_count}")
    print(f"  Failed    : {N_POINTS - converged_count}")
    print(f"  Total time: {elapsed:.1f}s  ({elapsed/N_POINTS:.2f}s/iter)")
    print("=" * 50)
    _log("DONE", f"Converged={converged_count}/{N_POINTS}  "
                 f"time={elapsed:.1f}s  avg={elapsed/N_POINTS:.2f}s/iter")
    idx.close()


if __name__ == "__main__":
    main()