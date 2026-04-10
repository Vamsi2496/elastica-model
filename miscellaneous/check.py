import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load test results
output_file = 'test_results_trimesh.npz'
with np.load(output_file) as data:
    loaded_points = data['test_points']
    loaded_mask = data['inside_mask']



# Extract ONLY inside points (vectorized - much faster!)
inside_points = loaded_points[loaded_mask]  # This filters for True values
#inside_points = loaded_points
phi1_inside = inside_points[:, 0]#*180/np.pi
phi2_inside = inside_points[:, 1]#*180/np.pi
d_inside = inside_points[:, 2]/400
#print(f"  d:    {np.sort(d_inside)}") 
print(f"Inside points: {len(inside_points)}/{len(loaded_points)}")

# Load surface mesh
CACHE_DIR = Path('delaunay_cache')
with np.load(CACHE_DIR / 'boundary_3d.npz') as f:
    boundary_3d = f['boundary_3d']
with np.load(CACHE_DIR / 'delaunay_surface.npz') as f:
    delaunay_surface_faces = f['surface_faces']

# Create 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')



# Plot INSIDE points as scatter
ax.scatter(phi1_inside, phi2_inside, d_inside, 
           #c='red', 
           marker='o', 
           s=20,
           alpha=0.6,
           label='Inside Points')


# Labels
ax.set_xlabel('phi1', labelpad=10)
ax.set_ylabel('phi2', labelpad=10)
ax.set_zlabel('d', rotation=0, labelpad=10)
ax.set_title(f'3D Bifurcation Diagram - Inside Points ({len(inside_points)} points)', pad=20)

# Set viewing angle
ax.view_init(elev=90, azim=0)
ax.legend()

plt.tight_layout()
plt.show()

