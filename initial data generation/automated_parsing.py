import numpy as np
import os
import h5py
from rtree import index
import glob

NDIM = 4   # number of state variables
NPAR = 9   # number of PAR values per solution record

def read_lines(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if ln:
            out.append(ln)
    return out


def count_sign_changes(arr):
    """Count the number of sign changes in an array."""
    if len(arr) < 2:
        return 0
    signs = np.sign(arr)
    # Remove zeros by replacing them with previous sign
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i-1]
    # Count sign changes
    return np.sum(np.diff(signs) != 0)


def parse_auto_s_file(fname, ndim, npar, phi_threshold=None):
    """
    Parse AUTO fort.8-style file (s.*).
    Mesh block: (1+NDIM) floats per line -> [t u1 u2 u3 u4]
    Derivative block: NDIM floats per line (skipped)
    PAR list: 9 floats split as 7 on one line + 2 on the next line
    """
    lines = read_lines(fname)
    blocks = []
    i = 0
    nlines = len(lines)

    while i < nlines:
        nums = np.fromstring(lines[i], sep=' ')
        # start of mesh block: (1 + NDIM) floats
        if nums.size == (1 + ndim):
            # --- mesh block ---
            t_list = []
            u1_list = []
            u2_list = []
            
            while i < nlines:
                nums = np.fromstring(lines[i], sep=' ')
                if nums.size != (1 + ndim):
                    break
                t_list.append(float(nums[0]))
                u1_list.append(float(nums[1]))
                u2_list.append(float(nums[2]))
                i += 1

            # --- derivative block: NDIM floats per line ---
            while i < nlines:
                nums = np.fromstring(lines[i], sep=' ')
                if nums.size != ndim:
                    break
                i += 1

            # --- find PAR: 7 floats on one line, 2 on the next ---
            par = []
            # skip non-float or wrong-length lines until we see 7 floats
            while i < nlines:
                nums = np.fromstring(lines[i], sep=' ')
                if nums.size == 7:
                    # this should be PAR(1..7)
                    par.extend([float(x) for x in nums])
                    i += 1
                    break
                else:
                    i += 1

            # now expect 2 more floats on the next line
            if i < nlines:
                nums = np.fromstring(lines[i], sep=' ')
                if nums.size >= 2:
                    par.append(float(nums[0]))
                    par.append(float(nums[1]))
                    i += 1

            if len(par) < npar:
                # incomplete PAR block -> stop
                break

            # compute derived quantities from PAR list
            d    = par[2]             # PAR(3)
            phi1 = par[4] - par[5]    # PAR(5)-PAR(6)
            phi2 = par[4] + par[5]    # PAR(5)+PAR(6)

            # Count sign changes in u2
            u2_array = np.array(u2_list, dtype=float)
            u2_sign_changes = count_sign_changes(u2_array)

            # Apply magnitude threshold filter
            if phi_threshold is None or (abs(phi1) <= phi_threshold and abs(phi2) <= phi_threshold):
                blocks.append({
                    "d": d,
                    "phi1": phi1,
                    "phi2": phi2,
                    "par": par,  # all 9 parameters
                    "u2_sign_changes": u2_sign_changes,
                    "t": np.array(t_list, dtype=float),
                    "u1": np.array(u1_list, dtype=float),
                    "u2": u2_array,
                })
        else:
            i += 1

    return blocks


def pack(blocks):
    d = []
    phi1 = []
    phi2 = []
    u2_sign_changes = []
    par_list = []
    t = []
    u1 = []
    u2 = []

    for b in blocks:
        d.append(b["d"])
        phi1.append(b["phi1"])
        phi2.append(b["phi2"])
        u2_sign_changes.append(b["u2_sign_changes"])
        par_list.append(b["par"])
        t.append(b["t"])
        u1.append(b["u1"])
        u2.append(b["u2"])

    d = np.array(d, dtype=float)
    phi1 = np.array(phi1, dtype=float)
    phi2 = np.array(phi2, dtype=float)
    u2_sign_changes = np.array(u2_sign_changes, dtype=int)
    par_array = np.array(par_list, dtype=float)  # shape: (n_blocks, 9)

    # variable-length arrays -> object arrays
    t = np.array(t, dtype=object)
    u1 = np.array(u1, dtype=object)
    u2 = np.array(u2, dtype=object)

    return d, phi1, phi2, u2_sign_changes, par_array, t, u1, u2


def process_folder(folder_path, hdf5_file, rtree_prefix, phi_threshold=None):
    """
    Process a single folder, save to HDF5, update index, then clear memory.
    Returns number of blocks processed.
    """
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_path}")
    print(f"{'='*60}")
    
    # Check if required files exist
    upper_file = os.path.join(folder_path, "s.upper")
    lower_file = os.path.join(folder_path, "s.lower")
    
    if not os.path.exists(upper_file) and not os.path.exists(lower_file):
        print(f"WARNING: Neither s.upper nor s.lower found in {folder_path}, skipping...")
        return 0
    
    blocks = []
    
    if os.path.exists(upper_file):
        blocks.extend(parse_auto_s_file(upper_file, NDIM, NPAR, phi_threshold))
        print(f"Parsed {len(blocks)} blocks from s.upper")
    else:
        print(f"WARNING: s.upper not found")
    
    prev_count = len(blocks)
    if os.path.exists(lower_file):
        blocks.extend(parse_auto_s_file(lower_file, NDIM, NPAR, phi_threshold))
        print(f"Parsed {len(blocks) - prev_count} blocks from s.lower")
    else:
        print(f"WARNING: s.lower not found")
    
    if len(blocks) == 0:
        print("No blocks parsed from this folder")
        return 0
    
    # Pack data
    d, phi1, phi2, u2_sign_changes, par_array, t, u1, u2 = pack(blocks)
    n_blocks = len(blocks)
    
    # Clear blocks list to free memory
    blocks.clear()
    del blocks
    
    # Append to HDF5 file
    start_idx = append_to_hdf5(hdf5_file, d, phi1, phi2, u2_sign_changes, 
                                par_array, t, u1)
    
    print(f"Saved {n_blocks} blocks from {os.path.basename(folder_path)}")
    
    # Update R-tree index immediately (incremental)
    update_rtree_index_hdf5(hdf5_file, rtree_prefix, start_idx)
    
    # Clear arrays to free memory
    del d, phi1, phi2, u2_sign_changes, par_array, t, u1, u2
    
    return n_blocks


def append_to_hdf5(filename, d, phi1, phi2, u2_sign_changes, par_array, t, u1):
    """
    Append data to HDF5 file. Returns starting index of new data.
    HDF5 only loads metadata, not entire arrays - very fast!
    """
    if os.path.exists(filename):
        # Append mode - only reads metadata, NOT the entire file
        with h5py.File(filename, 'a') as f:
            start_idx = len(f['d'])
            
            # Resize and append - NO loading of old data
            n_new = len(d)
            
            f['d'].resize((start_idx + n_new,))
            f['d'][start_idx:] = d
            
            f['phi1'].resize((start_idx + n_new,))
            f['phi1'][start_idx:] = phi1
            
            f['phi2'].resize((start_idx + n_new,))
            f['phi2'][start_idx:] = phi2
            
            f['inflection_points'].resize((start_idx + n_new,))
            f['inflection_points'][start_idx:] = u2_sign_changes
            
            f['parameters'].resize((start_idx + n_new, par_array.shape[1]))
            f['parameters'][start_idx:] = par_array
            
            # Variable-length arrays stored as vlen dtype
            f['t'].resize((start_idx + n_new,))
            f['t'][start_idx:] = t
            
            f['u1'].resize((start_idx + n_new,))
            f['u1'][start_idx:] = u1
            
        return start_idx
        
    else:
        # Create new file
        n = len(d)
        with h5py.File(filename, 'w') as f:
            # Create resizable datasets
            f.create_dataset('d', data=d, maxshape=(None,), chunks=True)
            f.create_dataset('phi1', data=phi1, maxshape=(None,), chunks=True)
            f.create_dataset('phi2', data=phi2, maxshape=(None,), chunks=True)
            f.create_dataset('inflection_points', data=u2_sign_changes, 
                           maxshape=(None,), chunks=True)
            f.create_dataset('parameters', data=par_array, 
                           maxshape=(None, par_array.shape[1]), chunks=True)
            
            # Variable-length arrays
            dt = h5py.vlen_dtype(np.dtype('float64'))
            f.create_dataset('t', shape=(n,), maxshape=(None,), chunks=True, dtype=dt)
            f.create_dataset('u1', shape=(n,), maxshape=(None,), chunks=True, dtype=dt)
            
            # Now write the data
            f['t'][:] = t
            f['u1'][:] = u1
            
        print(f"Created new HDF5 file: {filename}")
        return 0


def update_rtree_index_hdf5(hdf5_file, rtree_prefix, start_idx):
    """
    Update R-tree from HDF5 file (only reads needed data).
    Incrementally adds new points to existing index.
    """
    p = index.Property()
    p.dimension = 3
    
    with h5py.File(hdf5_file, 'r') as f:
        total_points = len(f['d'])
        # Only load NEW data
        d = f['d'][start_idx:]
        phi1 = f['phi1'][start_idx:]
        phi2 = f['phi2'][start_idx:]
    
    n_new = len(d)
    
    if n_new == 0:
        print("No new points to add to index")
        return
    
    if os.path.exists(f'{rtree_prefix}.dat'):
        # Append to existing index
        print(f"Updating R-tree index with {n_new} new points...")
        idx = index.Index(rtree_prefix, properties=p)
        
        for i, (d_val, phi1_val, phi2_val) in enumerate(zip(d, phi1, phi2), start=start_idx):
            bbox = (d_val, phi1_val, phi2_val, d_val, phi1_val, phi2_val)
            idx.insert(i, bbox, obj=i)
        
        idx.close()
        print(f"Index updated (total: {total_points} points)")
        
    else:
        # Create new index (first folder)
        print(f"Creating new R-tree index with {n_new} points...")
        
        def generator_function():
            for i in range(len(d)):
                bbox = (d[i], phi1[i], phi2[i], d[i], phi1[i], phi2[i])
                yield (start_idx + i, bbox, start_idx + i)
        
        idx = index.Index(rtree_prefix, generator_function(), properties=p)
        idx.close()
        print("R-tree index created")


def main():
    # Configuration
    HDF5_FILE = "auto_data_automatic.h5"
    RTREE_PREFIX = "auto_rtree_index_automatic"
    
    # Optional: phi threshold filter
    # PHI_THRESHOLD = 23.152235356
    PHI_THRESHOLD = None
    
    # Find all folders matching pattern x0p*
    folders = sorted(glob.glob("x0p*"))
    
    if len(folders) == 0:
        print("ERROR: No folders matching 'x0p*' pattern found!")
        print("Current directory:", os.getcwd())
        return
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING AUTO DATA")
    print(f"{'='*60}")
    print(f"Found {len(folders)} folders to process:")
    for folder in folders:
        print(f"  - {folder}")
    print(f"\nOutput files:")
    print(f"  - HDF5: {HDF5_FILE}")
    print(f"  - R-tree: {RTREE_PREFIX}.dat/.idx")
    
    # Process each folder one at a time
    total_blocks = 0
    for idx, folder in enumerate(folders, 1):
        print(f"\n[Folder {idx}/{len(folders)}]")
        
        if os.path.isdir(folder):
            n_blocks = process_folder(folder, HDF5_FILE, RTREE_PREFIX, PHI_THRESHOLD)
            total_blocks += n_blocks
        else:
            print(f"WARNING: {folder} is not a directory, skipping...")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Folders processed: {len(folders)}")
    print(f"Total blocks saved: {total_blocks}")
    
    if total_blocks > 0:
        # Print final statistics
        with h5py.File(HDF5_FILE, 'r') as f:
            n_points = len(f['d'])
            print(f"Final dataset size: {n_points} solution points")
            
            # Show memory usage info
            file_size_mb = os.path.getsize(HDF5_FILE) / (1024**2)
            print(f"HDF5 file size: {file_size_mb:.2f} MB")
    else:
        print("\nERROR: No blocks were processed!")


if __name__ == "__main__":
    main()
