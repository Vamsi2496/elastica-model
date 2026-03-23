import numpy as np
import os
import h5py
from rtree import index
import glob

NDIM = 4
NPAR = 9

__all__ = ["parse_folders","append_to_hdf5",]
def read_lines(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    return [ln.strip() for ln in lines if ln.strip()]


def count_sign_changes(arr):
    """
    counts the number of sign changes of an array

    Parameters
    ----------
    arr    : list[float]   an array of numerical data
    
    Returns
    -------
    number of sign changes: int
    """
    if len(arr) < 2:
        return 0
    signs = np.sign(arr)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    return int(np.sum(np.diff(signs) != 0))


def parse_auto_s_file(fname, ndim=NDIM, npar=NPAR):
    lines  = read_lines(fname)
    blocks = []
    i, nlines = 0, len(lines)

    while i < nlines:
        nums = np.fromstring(lines[i], sep=" ")
        if nums.size != (1 + ndim):
            i += 1
            continue

        t_l, u1_l, u2_l = [], [], []
        while i < nlines:
            nums = np.fromstring(lines[i], sep=" ")
            if nums.size != (1 + ndim):
                break
            t_l.append(nums[0]); u1_l.append(nums[1]); u2_l.append(nums[2])
            i += 1

        while i < nlines:                        # skip derivative block
            nums = np.fromstring(lines[i], sep=" ")
            if nums.size != ndim:
                break
            i += 1

        par = []
        while i < nlines:                        # find 7-float PAR line
            nums = np.fromstring(lines[i], sep=" ")
            if nums.size == 7:
                par.extend(nums.tolist()); i += 1; break
            i += 1
        if i < nlines:                           # 2 remaining PAR floats
            nums = np.fromstring(lines[i], sep=" ")
            if nums.size >= 2:
                par += [nums[0], nums[1]]; i += 1

        if len(par) < npar:
            break

        d    = par[2]
        phi1 = par[4] - par[5]
        phi2 = par[4] + par[5]
        u2_arr = np.array(u2_l, dtype=float)


        blocks.append({
                "d":               d,
                "phi1":            phi1,
                "phi2":            phi2,
                "par":             par,
                "u2_sign_changes": count_sign_changes(u2_arr),
                "t":               np.array(t_l,  dtype=float),
                "u1":              np.array(u1_l, dtype=float),
                "u2":              u2_arr,
        })
    return blocks


def pack(blocks):
    d, phi1, phi2, sc, par_list, t, u1, u2 = [], [], [], [], [], [], [], []
    for b in blocks:
        d.append(b["d"]);   phi1.append(b["phi1"]); phi2.append(b["phi2"])
        sc.append(b["u2_sign_changes"]); par_list.append(b["par"])
        t.append(b["t"]);   u1.append(b["u1"]);     u2.append(b["u2"])
    return (
        np.array(d,        dtype=float),
        np.array(phi1,     dtype=float),
        np.array(phi2,     dtype=float),
        np.array(sc,       dtype=int),
        np.array(par_list, dtype=float),
        np.array(t,        dtype=object),
        np.array(u1,       dtype=object),
        np.array(u2,       dtype=object),
    )


def append_to_hdf5(filename, d, phi1, phi2, sc, par_array, t, u1):
    """
    write/append the "auto" data to .h5 file 

    Parameters
    ----------
    filename  : (str)   file name to which data to be saved e.g. data.h5
    d         : list[float]
    phi1      : list[float]
    phi2      : list[float]
    sc        : list[int]   the number of inflection points
    par_array : list[list[float]] 9xn array of parameters
    t         : list[list[float]] 201xn array of arc length
    u1        : list[list[float]] 201xn array of theta
    
    Returns
    -------
    write/appennds the above data to a .h5 file
    s       : starting id of the .h5 file
    
    """
    if os.path.exists(filename):
        with h5py.File(filename, "a") as f:
            s = len(f["d"])
            n = len(d)
            for key, data in [("d", d), ("phi1", phi1),
                               ("phi2", phi2), ("inflection_points", sc)]:
                f[key].resize((s + n,)); f[key][s:] = data
            f["parameters"].resize((s + n, par_array.shape[1]))
            f["parameters"][s:] = par_array
            f["t"].resize((s + n,));  f["t"][s:]  = t
            f["u1"].resize((s + n,)); f["u1"][s:] = u1
        return s
    else:
        n  = len(d)
        dt = h5py.vlen_dtype(np.dtype("float64"))
        with h5py.File(filename, "w") as f:
            f.create_dataset("d",    data=d,    maxshape=(None,), chunks=True)
            f.create_dataset("phi1", data=phi1, maxshape=(None,), chunks=True)
            f.create_dataset("phi2", data=phi2, maxshape=(None,), chunks=True)
            f.create_dataset("inflection_points", data=sc,
                             maxshape=(None,), chunks=True)
            f.create_dataset("parameters", data=par_array,
                             maxshape=(None, par_array.shape[1]), chunks=True)
            f.create_dataset("t",  shape=(n,), maxshape=(None,),
                             chunks=True, dtype=dt)
            f.create_dataset("u1", shape=(n,), maxshape=(None,),
                             chunks=True, dtype=dt)
            f["t"][:] = t; f["u1"][:] = u1
        return 0


def update_rtree_index_hdf5(hdf5_file, rtree_prefix, start_idx):
    """
    create/update the rtree index file 

    Parameters
    ----------
    hdf5_file    : (str)   file name to whose data rtree need to be created/updated
    rtree_prefix : (str)   file name to which rtree index to be saved e.g. index
    start_idx    : (int)   index of .h5 data from which rtree index to be updated
    
    Returns
    -------
    create/update the rtree index
    """
    p = index.Property(); p.dimension = 3
    with h5py.File(hdf5_file, "r") as f:
        total = len(f["d"])
        d     = f["d"][start_idx:]
        phi1  = f["phi1"][start_idx:]
        phi2  = f["phi2"][start_idx:]
    n_new = len(d)
    if n_new == 0:
        return
    if os.path.exists(f"{rtree_prefix}.dat"):
        idx = index.Index(rtree_prefix, properties=p)
        for i, (dv, p1, p2) in enumerate(zip(d, phi1, phi2), start=start_idx):
            idx.insert(i, (dv, p1, p2, dv, p1, p2), obj=i)
        idx.close()
        print(f"  R-tree updated ({total} total points)")
    else:
        def _gen():
            for i in range(n_new):
                yield (start_idx + i,
                       (d[i], phi1[i], phi2[i], d[i], phi1[i], phi2[i]),
                       start_idx + i)
        idx = index.Index(rtree_prefix, _gen(), properties=p)
        idx.close()
        print(f"  R-tree created ({n_new} points)")


def process_folder(folder_path, hdf5_file="auto_data.h5",
                   rtree_prefix="auto_rtree_index"):
    """Parse s.upper / s.lower from one folder into HDF5 + R-tree."""
    upper = os.path.join(folder_path, "s.upper")
    lower = os.path.join(folder_path, "s.lower")
    if not os.path.exists(upper) and not os.path.exists(lower):
        print(f"  WARNING: no s.upper/s.lower in {folder_path}, skipping")
        return 0, [], [], [], []
    blocks = []
    if os.path.exists(upper):
        blocks.extend(parse_auto_s_file(upper, NDIM, NPAR))
    if os.path.exists(lower):
        blocks.extend(parse_auto_s_file(lower, NDIM, NPAR))
    if not blocks:
        return 0
    d, phi1, phi2, sc, par_array, t, u1, u2 = pack(blocks)
    n = len(blocks); blocks.clear()
    start_idx = append_to_hdf5(hdf5_file, d, phi1, phi2, sc, par_array, t, u1)
    update_rtree_index_hdf5(hdf5_file, rtree_prefix, start_idx)
        # ── Build explicit output lists ───────────────────────────
    hdf5_indices = []
    d_values     = []
    phi1_values  = []
    phi2_values  = []

    for j in range(n):
        hdf5_indices.append(start_idx + j)
        d_values.append(float(d[j]))
        phi1_values.append(float(phi1[j]))
        phi2_values.append(float(phi2[j]))
    del d, phi1, phi2, sc, par_array, t, u1, u2
    print(f"  ✓ {n} blocks  ← {os.path.basename(folder_path)}")
    return n, phi1_values, phi2_values, d_values, hdf5_indices


def parse_folders(folders, hdf5_file="auto_data.h5",
                  rtree_prefix="auto_rtree_index"):
    """
    Parse a list of folder paths into HDF5 + R-tree.

    Parameters
    ----------
    folders       : list[str]   absolute or relative folder paths (in this case d0p*)
    hdf5_file     : (str)   file name to which data generated by AUTO to be saved e.g. data.h5
    rtree_prefix  : (str)   file name to which rtree index to be saved e.g. index

    Returns
    -------
    total                : int   number of solution blocks saved
    phi1_values          : list[float]
    phi2_values          : list[float]
    d_values             : list[float]
    hdf5_indices         : list[float]
    """
    total = 0
    hdf5_indices = []
    d_values     = []
    phi1_values  = []
    phi2_values  = []
    print(f"\nParsing {len(folders)} folders...\n")
    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {folder}")
        if not os.path.isdir(folder):
            print(f"  WARNING: not a directory, skipping")
            continue

        n, f_phi1, f_phi2, f_d, f_idx  = process_folder(
            folder, hdf5_file, rtree_prefix
        )

        total        += n
        hdf5_indices += f_idx
        d_values     += f_d
        phi1_values  += f_phi1
        phi2_values  += f_phi2
    print(f"\n✓ Total blocks saved: {total}")
    print(f"✓ HDF5 : {hdf5_file}")
    print(f"✓ R-tree: {rtree_prefix}.dat/.idx")
    return total, phi1_values, phi2_values, d_values, hdf5_indices


def parse_all(pattern="d0p*", hdf5_file="auto_data.h5",
              rtree_prefix="auto_rtree_index"):
    """
    List out all folders matching a glob pattern (e.g. 'd0p*')
     
    Parse a list of folder paths into HDF5 + R-tree.

    Parameters
    ----------
    pattern:       : (str)  a pattern of folder names(in this case d0p*)
    hdf5_file     : (str)   file name to which data generated by AUTO to be saved e.g. data.h5
    rtree_prefix  : (str)   file name to which rtree index to be saved e.g. index

    Returns
    -------
    all matching folders

    """
    folders = sorted(f for f in glob.glob(pattern) if os.path.isdir(f))
    if not folders:
        print(f"No directories matching '{pattern}' found in {os.getcwd()}")
        return 0
    return parse_folders(folders, hdf5_file, rtree_prefix)
