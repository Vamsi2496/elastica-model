import numpy as np
import h5py
from rtree import index

# Configuration
d_min = 0.60
d_max = 0.99
d_step = 0.01
tolerance = 1e-4

# Initialize lists to store boundary data
all_d = []
all_phi1 = []
all_phi2 = []
all_par5 = []
all_par6 = []
all_indices = []

# Load R-tree index once
p = index.Property()
p.dimension = 3
idx = index.Index('auto_rtree_index', properties=p)

try:
    # ========== PART 1: Extract min/max for middle layers (0.61 to 0.98) ==========
    d_values = np.arange(d_min, d_max + d_step, d_step)
    
    for d_target in d_values:
        print(f"\nProcessing d = {d_target:.3f} (min/max extraction)")
        
        # Create query box for specific d value
        query_box = (d_target - tolerance,  # min_d
                     -1000,                  # min_phi1
                     -1000,                  # min_phi2
                     d_target + tolerance,   # max_d
                     1000,                   # max_phi1
                     1000)                   # max_phi2
        
        # Find all matching indices
        matches = list(idx.intersection(query_box, objects=True))
        match_indices = [hit.object for hit in matches]
        
        print(f"  Found {len(match_indices)} points at d = {d_target:.2f}")
        
        if len(match_indices) == 0:
            print("  No points found, skipping...")
            continue
        
        match_indices = sorted(match_indices)
        
        # Load data from HDF5
        with h5py.File("auto_data.h5", 'r') as f:
            phi1 = f['phi1'][match_indices]
            phi2 = f['phi2'][match_indices]
            par = f['parameters'][match_indices]
        
        par5 = par[:, 4]  # Extract Asymmetric
        par6 = par[:, 5]  # Extract Symmetric
        
        # Group by par(6) and find min/max par(5)
        unique_par6 = np.unique(par6)
        
        for p6_value in unique_par6:
            # Find all indices where par(6) equals current value
            mask = (par6 == p6_value)
            par5_group = par5[mask]
            phi1_group = phi1[mask]
            phi2_group = phi2[mask]
            indices_group = np.array(match_indices)[mask]
            
            # Find index of min par(5)
            min_idx = np.argmin(par5_group)
            all_d.append(d_target)
            all_phi1.append(phi1_group[min_idx])
            all_phi2.append(phi2_group[min_idx])
            all_par5.append(par5_group[min_idx])
            all_par6.append(p6_value)
            all_indices.append(indices_group[min_idx])
            
            # Find index of max par(5)
            max_idx = np.argmax(par5_group)
            all_d.append(d_target)
            all_phi1.append(phi1_group[max_idx])
            all_phi2.append(phi2_group[max_idx])
            all_par5.append(par5_group[max_idx])
            all_par6.append(p6_value)
            all_indices.append(indices_group[max_idx])
        
        print(f"  Extracted {len(unique_par6)} unique par(6) values -> {2*len(unique_par6)} boundary points")
    

finally:
    # CRITICAL: Always close the index
    idx.close()
    print("\n" + "="*60)
    print("R-tree index closed")
    print("="*60)

# FIRST: Convert lists to numpy arrays
all_d = np.array(all_d)
all_phi1 = np.array(all_phi1)
all_phi2 = np.array(all_phi2)
all_par5 = np.array(all_par5)
all_par6 = np.array(all_par6)
all_indices = np.array(all_indices)

print(f"\nTotal points before removing duplicates: {len(all_d)}")

# THEN: Remove duplicates (same rtree_index appearing multiple times)
unique_indices_mask = np.unique(all_indices, return_index=True)[1]
unique_indices_mask = np.sort(unique_indices_mask)


all_d = all_d[unique_indices_mask]
all_phi1 = all_phi1[unique_indices_mask]
all_phi2 = all_phi2[unique_indices_mask]
all_par5 = all_par5[unique_indices_mask]
all_par6 = all_par6[unique_indices_mask]
all_indices = all_indices[unique_indices_mask]

print(f"After removing duplicates: {len(all_d)} points")


# Save to NPZ file
output_filename = 'boundary_data_points.npz'
np.savez(output_filename,
         d=all_d,
         phi1=all_phi1,
         phi2=all_phi2,
         par5=all_par5,
         par6=all_par6,
         rtree_indices=all_indices)

print(f"\nTotal boundary points saved: {len(all_d)}")
print(f"Data saved to: {output_filename}")

# Stack arrays
data = np.column_stack([ all_phi1, all_phi2,  all_d*50 ])

xyz_file='boundary_data.xyz'

# Save with full precision (float64 = ~16-17 significant digits)
np.savetxt(xyz_file, data, 
           header=' phi1 phi2 d ',
           fmt='%.16e %.16e %.16e ',  # Scientific notation with 16 decimals
           comments='')

print(f"Data saved to: {xyz_file}")


# Print summary statistics
print(f"\nSummary Statistics:")
print(f"  d range: [{all_d.min():.3f}, {all_d.max():.3f}]")
print(f"  phi1 range: [{all_phi1.min():.3f}, {all_phi1.max():.3f}]")
print(f"  phi2 range: [{all_phi2.min():.3f}, {all_phi2.max():.3f}]")
print(f"  par(5) range: [{all_par5.min():.3f}, {all_par5.max():.3f}]")
print(f"  par(6) unique values: {len(np.unique(all_par6))}")

# Verification: Load and display saved data
print("\n" + "="*60)
print("Verification: Loading saved data")
print("="*60)
loaded_data = np.load(output_filename)
print(f"Keys in NPZ file: {list(loaded_data.keys())}")
print(f"Number of points: {len(loaded_data['d'])}")
loaded_data.close()
