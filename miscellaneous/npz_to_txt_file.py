import numpy as np

# Load NPZ file
with np.load('boundary_data_points.npz') as f:
    d = f['d']
    phi1 = f['phi1']
    phi2 = f['phi2']
    par5 = f['par5']
    par6 = f['par6']
    rtree_indices = f['rtree_indices']

# Stack arrays
data = np.column_stack([ phi1, phi2,  d*50 ])

# Save with full precision (float64 = ~16-17 significant digits)
np.savetxt('boundary_data.xyz', data, 
           header=' phi1 phi2 d ',
           fmt='%.16e %.16e %.16e ',  # Scientific notation with 16 decimals
           comments='')
