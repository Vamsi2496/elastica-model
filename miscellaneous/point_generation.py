import numpy as np
import time
from pathlib import Path

# ==================== OPTIMIZED RAY CASTING ====================

def precompute_face_bounds(vertices, faces):
    """
    Precompute bounding boxes for all faces.
    Returns array with min/max coordinates for each face.
    """
    print("  Precomputing face bounding boxes...")
    face_bounds = np.zeros((len(faces), 6))  # [x_min, x_max, y_min, y_max, z_min, z_max]
    
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        triangle = np.array([v0, v1, v2])
        
        face_bounds[i, 0] = triangle[:, 0].min()  # x_min
        face_bounds[i, 1] = triangle[:, 0].max()  # x_max
        face_bounds[i, 2] = triangle[:, 1].min()  # y_min
        face_bounds[i, 3] = triangle[:, 1].max()  # y_max
        face_bounds[i, 4] = triangle[:, 2].min()  # z_min
        face_bounds[i, 5] = triangle[:, 2].max()  # z_max
    
    print(f"  ✓ Precomputed bounds for {len(faces):,} faces")
    return face_bounds

def filter_faces_for_ray(point, ray_direction, face_bounds):
    """
    Filter faces that could possibly intersect with the ray.
    
    For ray along x-axis from point [px, py, pz]:
    - Only check faces with x_max >= px (in front of point)
    - Only check faces where py is within [y_min, y_max]
    - Only check faces where pz is within [z_min, z_max]
    """
    px, py, pz = point
    
    # Identify which axis the ray is primarily along
    abs_dir = np.abs(ray_direction)
    primary_axis = np.argmax(abs_dir)
    
    if primary_axis == 0:  # Ray along x-axis
        # Faces must be in front of point
        mask = face_bounds[:, 1] >= px  # x_max >= px
        
        # Face bounding box must contain the ray's y-z coordinates
        mask &= (face_bounds[:, 2] <= py) & (face_bounds[:, 3] >= py)  # y range
        mask &= (face_bounds[:, 4] <= pz) & (face_bounds[:, 5] >= pz)  # z range
        
    elif primary_axis == 1:  # Ray along y-axis
        mask = face_bounds[:, 3] >= py  # y_max >= py
        mask &= (face_bounds[:, 0] <= px) & (face_bounds[:, 1] >= px)  # x range
        mask &= (face_bounds[:, 4] <= pz) & (face_bounds[:, 5] >= pz)  # z range
        
    else:  # Ray along z-axis
        mask = face_bounds[:, 5] >= pz  # z_max >= pz
        mask &= (face_bounds[:, 0] <= px) & (face_bounds[:, 1] >= px)  # x range
        mask &= (face_bounds[:, 2] <= py) & (face_bounds[:, 3] >= py)  # y range
    
    # Return indices of faces that pass the filter
    candidate_face_indices = np.where(mask)[0]
    
    return candidate_face_indices

def ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2):
    """Möller–Trumbore ray-triangle intersection algorithm"""
    epsilon = 1e-8
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < epsilon:
        return False
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return False
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return False
    
    t = f * np.dot(edge2, q)
    return t > epsilon

def is_point_inside_surface_optimized(point, vertices, faces, face_bounds, ray_direction=None):
    """
    Optimized ray casting with face filtering.
    Only tests faces that could possibly intersect the ray.
    """
    if ray_direction is None:
        ray_direction = np.array([1.0, 0.0, 0.0])
    
    # Filter faces
    candidate_indices = filter_faces_for_ray(point, ray_direction, face_bounds)
    
    # Count intersections only with candidate faces
    intersection_count = 0
    for idx in candidate_indices:
        face = faces[idx]
        v0, v1, v2 = vertices[face]
        if ray_triangle_intersection(point, ray_direction, v0, v1, v2):
            intersection_count += 1
    
    return intersection_count % 2 == 1, len(candidate_indices)

def test_points_optimized(points, vertices, faces, ray_direction=None, show_progress=True):
    """
    Test multiple points with optimized ray casting.
    
    Parameters:
    -----------
    points : array (N, 3)
        Points to test
    vertices : array (M, 3)
        Surface mesh vertices
    faces : array (K, 3)
        Surface mesh face indices
    ray_direction : array (3,), optional
        Ray direction (default: [1, 0, 0])
    show_progress : bool
        Print progress updates
    
    Returns:
    --------
    inside_mask : boolean array (N,)
        True if point is inside surface
    stats : dict
        Performance statistics
    """
    if ray_direction is None:
        ray_direction = np.array([1.0, 0.0, 0.0])
    
    n_points = len(points)
    inside_mask = np.zeros(n_points, dtype=bool)
    
    # Precompute face bounds (only once)
    face_bounds = precompute_face_bounds(vertices, faces)
    
    print(f"\n  Testing {n_points:,} points with optimized ray casting...")
    t_start = time.time()
    
    total_faces_checked = 0
    
    for i, point in enumerate(points):
        is_inside, n_candidates = is_point_inside_surface_optimized(
            point, vertices, faces, face_bounds, ray_direction
        )
        inside_mask[i] = is_inside
        total_faces_checked += n_candidates
        
        if show_progress and (i + 1) % 10000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (n_points - i - 1) / rate if rate > 0 else 0
            avg_faces = total_faces_checked / (i + 1)
            print(f"    Progress: {i+1:,}/{n_points:,} | Rate: {rate:.1f} pts/s | "
                  f"Avg faces/point: {avg_faces:.0f} | ETA: {remaining:.1f}s")
    
    t_total = time.time() - t_start
    
    stats = {
        'total_time': t_total,
        'rate': n_points / t_total if t_total > 0 else 0,
        'total_faces': len(faces),
        'avg_faces_checked': total_faces_checked / n_points,
        'reduction_factor': len(faces) / (total_faces_checked / n_points) if total_faces_checked > 0 else 1
    }
    
    print(f"\n  ✓ Completed in {t_total:.2f}s ({stats['rate']:.1f} points/sec)")
    print(f"  Optimization: Checked {stats['avg_faces_checked']:.0f}/{stats['total_faces']:,} "
          f"faces per point ({stats['reduction_factor']:.1f}x speedup)")
    
    return inside_mask, stats

# ==================== USAGE ====================

if __name__ == "__main__":
    
    print("="*60)
    print("OPTIMIZED RAY CASTING TEST")
    print("="*60)
    
    # Load surface mesh
    print("\nLoading surface mesh...")
    CACHE_DIR = Path('delaunay_cache')
    
    with np.load(CACHE_DIR / 'boundary_3d.npz') as f:
        boundary_3d = f['boundary_3d']
    with np.load(CACHE_DIR / 'delaunay_surface.npz') as f:
        delaunay_surface_faces = f['surface_faces']
    
    print(f"  Vertices: {len(boundary_3d):,}")
    print(f"  Faces: {len(delaunay_surface_faces):,}")
    
    # Generate test points
    phi1_min, phi1_max = boundary_3d[:, 0].min(), boundary_3d[:, 0].max()
    phi2_min, phi2_max = boundary_3d[:, 1].min(), boundary_3d[:, 1].max()
    d_min, d_max = boundary_3d[:, 2].min(), boundary_3d[:, 2].max()
    
    print(f"\nBounding box:")
    print(f"  phi1: [{phi1_min:.2f}, {phi1_max:.2f}]")
    print(f"  phi2: [{phi2_min:.2f}, {phi2_max:.2f}]")
    print(f"  d:    [{d_min:.4f}, {d_max:.4f}]")

    code_start = time.time() 
    
    n_test = 1000000  # Adjust as needed
    test_points = np.random.uniform(
        [phi1_min, phi2_min, d_min],
        [phi1_max, phi2_max, d_max],
        size=(n_test, 3)
    )
    #test_points = np.array([[-55, 0, 0.81]])
    print(f"\nGenerated {n_test:,} random test points")
    
    # Test points with optimized method
    inside_mask, stats = test_points_optimized(
        test_points, boundary_3d, delaunay_surface_faces,
        ray_direction=np.array([1.0, 0.0, 0.0]),
        show_progress=True
    )
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"Points inside: {np.sum(inside_mask):,}/{n_test:,} ({np.sum(inside_mask)/n_test*100:.1f}%)")
    print(f"Points outside: {n_test - np.sum(inside_mask):,}/{n_test:,}")
    
    # Print sample test points
    print(f"\n{'='*60}")
    print("SAMPLE TEST POINTS")
    print('='*60)
    # Save to .npz file
    output_file = 'test_results.npz'
    np.savez_compressed(output_file,
         test_points=test_points,
         inside_mask=inside_mask)
    # Show first 5 points
    print("\nFirst 5 test points:")
    #for i in range(n_test):
    for i in range(min(5, n_test)):
        status = "INSIDE" if inside_mask[i] else "OUTSIDE"
        print(f"  Point {i}: phi1={test_points[i,0]:7.2f}, phi2={test_points[i,1]:7.2f}, "
              f"d={test_points[i,2]:.4f} → {status}")
    
    code_end = time.time()
    total_time = code_end - code_start
    print(f"\n  code start time: {code_start:.2f}s; code end time: {code_end:.2f}s code run time: {total_time/60:.2f}minutes ")

