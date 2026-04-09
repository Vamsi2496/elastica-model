import os
import h5py
import numpy as np
from rtree import index

# Check if index exists
if os.path.exists('auto_rtree_index.dat') and os.path.exists('auto_rtree_index.idx'):
    # Load existing index
    p = index.Property()
    p.dimension = 3
    idx = index.Index('auto_rtree_index', properties=p)
    print("Index loaded from disk")
else:
    print("Error: Index files not found! Run index creation script first.")
    exit()

# Search point
search_d = 0.97 
search_phi1 = -20
search_phi2 = 10

# Define search radius for each dimension
radius_d = 0.005      # Adjust based on your data density
radius_phi1 = 0.4
radius_phi2 = 0.4

# Create bounding box for intersection query
query_bbox = (search_d - radius_d, search_phi1 - radius_phi1, search_phi2 - radius_phi2,
              search_d + radius_d, search_phi1 + radius_phi1, search_phi2 + radius_phi2)

# Find all points in intersection region
intersections = list(idx.intersection(query_bbox, objects=True))
print(f"Found {len(intersections)} points in search region")

if len(intersections) == 0:
    print("No points found in search region. Try increasing radius values.")
    idx.close()
    exit()

# Load data and find closest point
with h5py.File("auto_data.h5", 'r') as f:
    min_distance = float('inf')
    closest_idx = None
    closest_coords = None
    
    for hit in intersections:
        i = hit.object
        d_val = f['d'][i]
        phi1_val = f['phi1'][i]
        phi2_val = f['phi2'][i]
        
        # Calculate Euclidean distance (normalized by typical scales)
        distance = np.sqrt(((d_val - search_d) / radius_d)**2 + 
                          ((phi1_val - search_phi1) / radius_phi1)**2 + 
                          ((phi2_val - search_phi2) / radius_phi2)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
            closest_coords = (d_val, phi1_val, phi2_val)
    
    # Display results for closest point
    print(f"\nClosest point found:")
    print(f"Index={closest_idx}")
    print(f"d={closest_coords[0]}, phi1={closest_coords[1]}, phi2={closest_coords[2]}")
    #print(f"Normalized distance={min_distance:.6f}")
    print(f"parameters={f['parameters'][closest_idx]}")
    print(f"Inflection points={f['inflection_points'][closest_idx]}")
    
    # Load solution arrays for this match
    t_vals = f['t'][closest_idx]
    u1_vals = f['u1'][closest_idx]
    print(f"Solution array shapes: t={t_vals.shape}, u1={u1_vals.shape}")

idx.close()
print("\nR-tree index closed")
