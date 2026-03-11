
# master_loop_parallel.py  (Python 3)
#
# Architecture:
#   - ThreadPoolExecutor with N_WORKERS threads
#   - Each worker runs in its own sub-directory -> no file conflicts
#   - Workers share mesh copies (one per worker) and R-tree (locked reads)
#   - After every batch: idx.close() -> HDF5 flush -> R-tree insert -> idx.reopen()
#   - h5py: all reads AND writes protected by hdf5_lock
#   - R-tree: all reads AND writes protected by rtree_lock

import concurrent.futures
import threading
import shutil
import subprocess
import json
import os
import time
import numpy as np
import trimesh
import h5py
from rtree import index as rtree_index


# ===============================================================================
#  CONFIG
# ===============================================================================
N_POINTS      = 160000
N_WORKERS     = 6

EQ_DIR_BASE   = r"C:\Users\sanch\Desktop\Restart solution"
MESH_FILE     = "final_mesh.off"
HDF5_FILE     = "auto_data.h5"
RTREE_BASE    = "auto_rtree_index"
DAT_FILE      = "s.initial"
LOG_FILE      = "pipeline_log.txt"
BRIDGE_EXE    = r"C:\Python26\python.exe"
BRIDGE_SCRIPT = "bridge.py"

D_SCALE      = 50          # normalised d = raw_d / D_SCALE
SAMPLE_BATCH = 5
NDIM         = 4
NPAR         = 9
RADIUS_D     = 0.01
RADIUS_PHI1  = 2.0
RADIUS_PHI2  = 2.0

WORKER_DIRS = [os.path.join(EQ_DIR_BASE, f"worker_{i}") for i in range(N_WORKERS)]

# Locks — protect ALL accesses to shared C-level libraries
hdf5_lock  = threading.Lock()   # HDF5 C library has global state
rtree_lock = threading.Lock()   # libspatialindex is not thread-safe
log_lock   = threading.Lock()


# ===============================================================================
#  SETUP
# ===============================================================================

def setup_worker_dirs():
    template_dir = os.path.join(EQ_DIR_BASE, "template")
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}\n"
            f"Create it and place s.initial, restart.auto, el3.* etc. inside."
        )
    for wdir in WORKER_DIRS:
        os.makedirs(wdir, exist_ok=True)
        for fname in os.listdir(template_dir):
            src = os.path.join(template_dir, fname)
            dst = os.path.join(wdir, fname)
            if os.path.isdir(src):
                continue
            if os.path.getsize(src) > 50 * 1024 * 1024:
                continue
            shutil.copy2(src, dst)
    print(f"Worker dirs ready ({len(WORKER_DIRS)} dirs from template)")


def patch_bridge_for_env():
    """Rewrite bridge.py once so eq_dir is read from AUTO_EQ_DIR env-var."""
    bridge_path = os.path.join(EQ_DIR_BASE, BRIDGE_SCRIPT)
    with open(bridge_path, 'r') as f:
        src = f.read()
    old = 'eq_dir = "C:\\\\Users\\\\sanch\\\\Desktop\\\\Restart solution"'
    new = ('eq_dir = __import__("os").environ.get("AUTO_EQ_DIR",'
           ' "C:\\\\Users\\\\sanch\\\\Desktop\\\\Restart solution")')
    if old in src:
        with open(bridge_path, 'w') as f:
            f.write(src.replace(old, new))
        print("bridge.py patched to read AUTO_EQ_DIR")
    else:
        print("bridge.py already patched — skipping")


# ===============================================================================
#  ONE-TIME LOADERS
# ===============================================================================

def load_mesh():
    print("Loading surface mesh...")
    raw  = trimesh.load(MESH_FILE, force='mesh')
    mesh = trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces)
    # mesh.use_embree = True   # uncomment if pyembree installed
    print(f"  Watertight: {mesh.is_watertight},  Volume: {mesh.volume:.4f}")
    return mesh


def load_rtree():
    p = rtree_index.Property()
    p.dimension = 3
    exists = os.path.exists(f"{RTREE_BASE}.dat")
    print(f"  {'Loading' if exists else 'Creating'} R-tree index...")
    return rtree_index.Index(RTREE_BASE, properties=p)


# ===============================================================================
#  STEP 1 — sample, nearest lookup, write s.initial
# ===============================================================================

def _sample_inside(mesh_copy):
    """Sample random points until one lands inside the mesh."""
    verts = mesh_copy.vertices
    lo = np.array([verts[:, 0].min(), verts[:, 1].min(), 0.60 * D_SCALE])
    hi = np.array([verts[:, 0].max(), verts[:, 1].max(), 0.99 * D_SCALE])
    found, n_tested = None, 0
    while found is None:
        batch     = np.random.uniform(lo, hi, size=(SAMPLE_BATCH, 3))
        mask      = mesh_copy.contains(batch)
        n_tested += SAMPLE_BATCH
        if mask.any():
            found = batch[np.argmax(mask)]
    return found, n_tested


def _find_nearest(found_point, idx):
    """R-tree search + HDF5 lookup for nearest stored solution."""
    s_phi1, s_phi2 = found_point[0], found_point[1]
    s_d = found_point[2] / D_SCALE

    query_bbox = (
        s_d     ,    s_phi1 - RADIUS_PHI1,  s_phi2 - RADIUS_PHI2,
        s_d     + RADIUS_D,    s_phi1 + RADIUS_PHI1,  s_phi2 + RADIUS_PHI2,
    )

    # libspatialindex is NOT thread-safe for concurrent reads — lock required
    with rtree_lock:
        hits = list(idx.intersection(query_bbox, objects=True))
    if not hits:
        return None

    # HDF5 C library has global state — lock required even for reads
    with hdf5_lock:
        with h5py.File(HDF5_FILE, 'r') as f:
            best_dist, best_i = float('inf'), None
            for hit in hits:
                i = hit.object
                dist = np.sqrt(
                    ((f['d'][i]    - s_d)    / RADIUS_D   ) ** 2 +
                    ((f['phi1'][i] - s_phi1) / RADIUS_PHI1) ** 2 +
                    ((f['phi2'][i] - s_phi2) / RADIUS_PHI2) ** 2
                )
                if dist < best_dist:
                    best_dist, best_i = dist, i
            params  = f['parameters'][best_i]
            t_vals  = f['t'][best_i]
            u1_vals = f['u1'][best_i]
    return params, t_vals, u1_vals


def _build_arrays(t_vals, u1_vals):
    """Compute u2 = du1/dt, u3 = integral(cos u1), u4 = integral(sin u1)."""
    u2 = np.gradient(u1_vals, t_vals)
    u3, u4 = np.zeros_like(u1_vals), np.zeros_like(u1_vals)
    if len(t_vals) > 1:
        dt = np.diff(t_vals)
        u3[1:] = np.cumsum(0.5 * (np.cos(u1_vals[:-1]) + np.cos(u1_vals[1:])) * dt)
        u4[1:] = np.cumsum(0.5 * (np.sin(u1_vals[:-1]) + np.sin(u1_vals[1:])) * dt)
    return u2, u3, u4


def _fmt_sol(t, u1, u2, u3, u4):
    return f" {t: .10E}   {u1: .10E}   {u2: .10E}   {u3: .10E}   {u4: .10E}\n"


def _fmt_par(values):
    return "   ".join(f"{v: .10E}" for v in values) + "\n"


def _write_s_initial(worker_dir, params, t_vals, u1_vals, u2, u3, u4):
    dat_path = os.path.join(worker_dir, DAT_FILE)
    with open(dat_path, 'r') as f:
        lines = f.readlines()

    sol_lines = [_fmt_sol(t, u1, u2_, u3_, u4_)
                 for t, u1, u2_, u3_, u4_ in zip(t_vals, u1_vals, u2, u3, u4)]
    n = len(sol_lines)
    lines[1 : 1 + n] = sol_lines

    # Robustly locate the parameter block (first 7-float line after solution)
    param_start = None
    for k in range(1 + n, len(lines)):
        vals = lines[k].split()
        if len(vals) == 7:
            try:
                list(map(float, vals))
                param_start = k
                break
            except ValueError:
                pass
    if param_start is not None:
        lines[param_start]     = _fmt_par(params[:7])
        lines[param_start + 1] = _fmt_par(params[7:])
    else:
        print(f"  [WARN] param block not found in {dat_path}")

    with open(dat_path, 'w') as f:
        f.writelines(lines)


# ===============================================================================
#  STEP 3 — parse AUTO output (worker-side, returns blocks only)
# ===============================================================================

def _sign_changes(arr):
    if len(arr) < 2:
        return 0
    s = np.sign(arr)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    return int(np.sum(np.diff(s) != 0))


def _parse_s_file(fname):
    with open(fname, 'r') as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    blocks, i, N = [], 0, len(raw)

    while i < N:
        nums = np.fromstring(raw[i], sep=' ')
        if nums.size != (1 + NDIM):
            i += 1
            continue

        t_l, u1_l, u2_l = [], [], []
        while i < N:
            nums = np.fromstring(raw[i], sep=' ')
            if nums.size != (1 + NDIM):
                break
            t_l.append(nums[0]); u1_l.append(nums[1]); u2_l.append(nums[2])
            i += 1

        while i < N:                            # skip derivative block
            nums = np.fromstring(raw[i], sep=' ')
            if nums.size != NDIM:
                break
            i += 1

        par = []
        while i < N:
            nums = np.fromstring(raw[i], sep=' ')
            if nums.size == 7:
                par.extend(nums.tolist()); i += 1; break
            i += 1
        if i < N:
            nums = np.fromstring(raw[i], sep=' ')
            if nums.size >= 2:
                par += [nums[0], nums[1]]; i += 1

        if len(par) < NPAR:
            break

        u2_arr = np.array(u2_l, dtype=float)
        blocks.append({
            'd':    par[2],
            'phi1': par[4] - par[5],
            'phi2': par[4] + par[5],
            'par':  par,
            'sc':   _sign_changes(u2_arr),
            't':    np.array(t_l,  dtype=float),
            'u1':   np.array(u1_l, dtype=float),
        })
    return blocks


# ===============================================================================
#  BATCH FLUSH — main thread only, idx already closed before this is called
# ===============================================================================

def _batch_flush(all_blocks):
    """
    Write all collected blocks to HDF5, then open a fresh R-tree handle,
    insert all new entries, and close it.
    Called from main() only after idx.close() — zero concurrent access.
    """
    if not all_blocks:
        return

    d    = np.array([b['d']    for b in all_blocks], dtype=float)
    phi1 = np.array([b['phi1'] for b in all_blocks], dtype=float)
    phi2 = np.array([b['phi2'] for b in all_blocks], dtype=float)
    sc   = np.array([b['sc']   for b in all_blocks], dtype=int)
    par  = np.array([b['par']  for b in all_blocks], dtype=float)
    t_a  = np.array([b['t']    for b in all_blocks], dtype=object)
    u1_a = np.array([b['u1']   for b in all_blocks], dtype=object)
    n    = len(d)

    # ── HDF5 append ────────────────────────────────────────────────────────────
    with hdf5_lock:
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
            start_idx = s
        else:
            dt_vlen = h5py.vlen_dtype(np.dtype('float64'))
            with h5py.File(HDF5_FILE, 'w') as f:
                f.create_dataset('d',    data=d,   maxshape=(None,), chunks=True)
                f.create_dataset('phi1', data=phi1, maxshape=(None,), chunks=True)
                f.create_dataset('phi2', data=phi2, maxshape=(None,), chunks=True)
                f.create_dataset('inflection_points', data=sc,
                                 maxshape=(None,), chunks=True)
                f.create_dataset('parameters', data=par,
                                 maxshape=(None, par.shape[1]), chunks=True)
                dt = h5py.vlen_dtype(np.dtype('float64'))
                f.create_dataset('t',  shape=(n,), maxshape=(None,),
                                 chunks=True, dtype=dt)
                f.create_dataset('u1', shape=(n,), maxshape=(None,),
                                 chunks=True, dtype=dt)
                f['t'][:] = t_a;  f['u1'][:] = u1_a
            start_idx = 0
        print(f"  HDF5: +{n} blocks (total {start_idx + n})")

    # ── R-tree: open fresh handle, insert, close ───────────────────────────────
    # No lock needed here — idx was closed in main() before this call
    p = rtree_index.Property()
    p.dimension = 3
    flush_idx = rtree_index.Index(RTREE_BASE, properties=p)
    for i, (dv, p1, p2) in enumerate(zip(d, phi1, phi2), start=start_idx):
        flush_idx.insert(i, (dv, p1, p2, dv, p1, p2), obj=i)
    flush_idx.close()
    print(f"  R-tree: +{n} entries inserted, index closed")


# ===============================================================================
#  WORKER — one full iteration in its own directory
# ===============================================================================

def _worker(worker_id, iteration_id, mesh_copy, idx):
    """
    Each worker gets:
      - its own mesh_copy  -> no shared mutable trimesh state
      - shared idx         -> reads protected by rtree_lock
    Returns a result dict; never writes to HDF5 or R-tree directly.
    """
    t0       = time.time()
    wdir     = WORKER_DIRS[worker_id]
    sentinel = os.path.join(wdir, "auto_status.json")

    def _fail(msg):
        return {'iter': iteration_id, 'wid': worker_id, 'converged': False,
                'blocks': None, 'elapsed': time.time() - t0, 'msg': msg}

    # Step 1a — sample a point inside the mesh
    try:
        found_point, n_tested = _sample_inside(mesh_copy)
    except Exception as e:
        return _fail(f"sampling failed: {e}")

    # Step 1b — nearest HDF5 lookup (both locks held inside _find_nearest)
    try:
        result = _find_nearest(found_point, idx)
    except Exception as e:
        return _fail(f"HDF5 lookup failed: {e}")
    if result is None:
        return _fail("no R-tree hit — increase RADIUS_*")

    params, t_vals, u1_vals = result

    # Step 1c — build arrays and write worker-local s.initial
    try:
        u2, u3, u4 = _build_arrays(t_vals, u1_vals)
        _write_s_initial(wdir, params, t_vals, u1_vals, u2, u3, u4)
        # Save normalised d so restart.auto never needs to know D_SCALE
        np.savez_compressed(
            os.path.join(wdir, 'target_point.npz'),
            target_point=np.array([
                found_point[0],             # phi1 unchanged
                found_point[1],             # phi2 unchanged
                found_point[2] / D_SCALE    # d already normalised
            ])
        )
    except Exception as e:
        return _fail(f"write s.initial failed: {e}")

    # Step 2 — run AUTO (subprocess releases GIL entirely)
    env = os.environ.copy()
    env['AUTO_EQ_DIR'] = wdir
    ret = subprocess.call(
        [BRIDGE_EXE, os.path.join(EQ_DIR_BASE, BRIDGE_SCRIPT)],
        env=env, cwd=wdir
    )
    if ret != 0:
        return _fail(f"bridge.py exited {ret}")

    # Step 3 — read convergence sentinel
    try:
        with open(sentinel) as f:
            converged = json.load(f)["converged"]
    except Exception as e:
        return _fail(f"sentinel read failed: {e}")

    if not converged:
        return {'iter': iteration_id, 'wid': worker_id, 'converged': False,
                'blocks': None, 'elapsed': time.time() - t0,
                'msg': 'NOT CONVERGED'}

    # Step 4 — parse AUTO output (no HDF5/R-tree writes here)
    try:
        blocks = _parse_s_file(os.path.join(wdir, "s.curr_data"))
    except Exception as e:
        return {'iter': iteration_id, 'wid': worker_id, 'converged': True,
                'blocks': None, 'elapsed': time.time() - t0,
                'msg': f"CONVERGED but parse failed: {e}"}

    return {'iter': iteration_id, 'wid': worker_id, 'converged': True,
            'blocks': blocks, 'elapsed': time.time() - t0,
            'msg': f"SUCCESS ({n_tested} pts sampled, {len(blocks)} blocks)"}


# ===============================================================================
#  LOGGING
# ===============================================================================

def _log(msg):
    with log_lock:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")


# ===============================================================================
#  MAIN
# ===============================================================================

def main():
    t_total = time.time()

    setup_worker_dirs()
    patch_bridge_for_env()

    # Load one mesh copy per worker — trimesh BVH is not thread-safe
    print("Loading mesh copies...")
    base_mesh = load_mesh()
    mesh_copies = [
        trimesh.Trimesh(
            vertices=base_mesh.vertices.copy(),
            faces=base_mesh.faces.copy()
        )
        for _ in WORKER_DIRS
    ]
    print(f"  {len(mesh_copies)} independent mesh copies ready")

    print("Loading R-tree...")
    idx = load_rtree()

    with open(LOG_FILE, 'w') as f:
        f.write(f"Pipeline started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"N_POINTS={N_POINTS}  N_WORKERS={N_WORKERS}\n"
                + "=" * 60 + "\n")

    converged_total = failed_total = 0
    iteration = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        while iteration < N_POINTS:
            batch_size = min(N_WORKERS, N_POINTS - iteration)

            # Submit batch — idx is open for reads, each worker uses its own mesh copy
            futures = {
                pool.submit(
                    _worker,
                    wid,
                    iteration + wid,
                    mesh_copies[wid],   # private mesh copy per worker
                    idx
                ): wid
                for wid in range(batch_size)
            }
            iteration += batch_size

            # Collect all results (blocks held in memory, not yet written)
            all_new_blocks = []
            batch_converged = 0
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                tag = "✓" if r['converged'] else "✗"
                print(f"  [{tag}] iter={r['iter']:4d}  wid={r['wid']}"
                      f"  {r['elapsed']:.2f}s  {r['msg']}")
                _log(f"iter={r['iter']} wid={r['wid']} {r['msg']}")

                if r['converged']:
                    converged_total += 1
                    batch_converged += 1
                    if r['blocks']:
                        all_new_blocks.extend(r['blocks'])
                else:
                    failed_total += 1

            # Flush: close idx -> write HDF5 + R-tree -> reopen idx
            if all_new_blocks:
                print(f"\n  Flushing {len(all_new_blocks)} blocks "
                      f"from {batch_converged} converged workers...")

                idx.close()
                print("  R-tree closed")

                _batch_flush(all_new_blocks)    # HDF5 write + R-tree insert + close

                idx = load_rtree()              # fresh handle for next batch
                print("  R-tree reopened")

            print(f"  Batch done | {iteration}/{N_POINTS} "
                  f"| converged so far: {converged_total}\n")

    elapsed = time.time() - t_total
    idx.close()

    summary = (f"\nPipeline complete\n"
               f"  Total     : {N_POINTS}\n"
               f"  Converged : {converged_total}\n"
               f"  Failed    : {failed_total}\n"
               f"  Total time: {elapsed:.1f}s\n"
               f"  Avg/iter  : {elapsed/N_POINTS:.2f}s  "
               f"(effective {elapsed/(N_POINTS/N_WORKERS):.2f}s/batch)")
    print(summary)
    _log(summary)


if __name__ == "__main__":
    main()
