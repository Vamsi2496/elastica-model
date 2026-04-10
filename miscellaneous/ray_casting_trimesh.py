import numpy as np
import time
from pathlib import Path
import trimesh

# ==================== TRIMESH RAY CASTING ====================

def test_points_trimesh(points, vertices, faces, show_progress=True):
    """
    Test multiple points using trimesh's built-in ray casting.

    Parameters:
    -----------
    points : array (N, 3)
        Points to test
    vertices : array (M, 3)
        Surface mesh vertices
    faces : array (K, 3)
        Surface mesh face indices
    show_progress : bool
        Print progress updates

    Returns:
    --------
    inside_mask : boolean array (N,)
        True if point is inside surface
    stats : dict
        Performance statistics
    """
    n_points = len(points)

    print(f"\n  Creating trimesh object...")
    t_mesh_start = time.time()

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    t_mesh_end = time.time()
    print(f"  ✓ Mesh created in {t_mesh_end - t_mesh_start:.2f}s")
    print(f"  Mesh info: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print(f"  Watertight: {mesh.is_watertight}, Volume: {mesh.volume:.4f}")

    print(f"\n  Testing {n_points:,} points with trimesh ray casting...")
    t_start = time.time()

    # Use trimesh's contains_points method (uses ray casting internally)
    inside_mask = mesh.contains(points)

    t_total = time.time() - t_start

    stats = {
        'total_time': t_total,
        'mesh_creation_time': t_mesh_end - t_mesh_start,
        'ray_casting_time': t_total,
        'rate': n_points / t_total if t_total > 0 else 0,
        'total_faces': len(faces),
        'is_watertight': mesh.is_watertight
    }

    print(f"\n  ✓ Completed in {t_total:.2f}s ({stats['rate']:.1f} points/sec)")

    return inside_mask, stats



# ==================== USAGE ====================

if __name__ == "__main__":

    print("="*60)
    print("TRIMESH RAY CASTING TEST")
    print("="*60)

    # Load surface mesh
    print("\nLoading surface mesh...")
    CACHE_DIR = Path('delaunay_cache')

    #with np.load(CACHE_DIR / 'boundary_3d.npz') as f:
     #   boundary_3d = f['boundary_3d']
#    with np.load(CACHE_DIR / 'delaunay_surface.npz') as f:
 #       delaunay_surface_faces = f['surface_faces']

    mesh = trimesh.load("surface mesh scaled 400 alpha 2.ply", force='mesh')
    #mesh = trimesh.load("surface mesh closed holes.ply", force='mesh')
    boundary_3d = mesh.vertices
    delaunay_surface_faces = mesh.faces




    print(f"  Vertices: {len(boundary_3d):,}")
    print(f"  Faces: {len(delaunay_surface_faces):,}")

    # Generate test points
    phi1_min, phi1_max = boundary_3d[:, 0].min(), boundary_3d[:, 0].max()
    phi2_min, phi2_max = boundary_3d[:, 1].min(), boundary_3d[:, 1].max()
    d_min, d_max = boundary_3d[:, 2].min(), boundary_3d[:, 2].max()

    print(f"\nBounding box:")
    print(f"  phi1: [{phi1_min:.3f}, {phi1_max:.3f}]")
    print(f"  phi2: [{phi2_min:.3f}, {phi2_max:.3f}]")
    #print(f"  phi1: [{phi1_min*180/np.pi:.3f}, {phi1_max*180/np.pi:.3f}]")
    #print(f"  phi2: [{phi2_min*180/np.pi:.3f}, {phi2_max*180/np.pi:.3f}]")
    print(f"  d:    [{d_min/400:.4f}, {d_max/400:.4f}]")

    code_start = time.time() 

    n_test = 1000  # Adjust as needed
    test_points = np.random.uniform(
        [phi1_min, phi2_min, d_min],
        [phi1_max, phi2_max, d_max],
        size=(n_test, 3)
    )
    print(f"\nGenerated {n_test:,} random test points")

    # METHOD 1: Use trimesh's built-in contains method (RECOMMENDED - FASTEST)
    print("\n" + "="*60)
    print("METHOD 1: trimesh.contains() [RECOMMENDED]")
    print("="*60)
    inside_mask, stats = test_points_trimesh(
        test_points, boundary_3d, delaunay_surface_faces,
        show_progress=True
    )

    # Optionally test METHOD 2 for comparison
    # Uncomment below to compare with explicit ray intersection method
    """
    print("\n" + "="*60)
    print("METHOD 2: Explicit ray intersections")
    print("="*60)
    inside_mask_alt, stats_alt = test_points_trimesh_ray_method(
        test_points, boundary_3d, delaunay_surface_faces,
        ray_direction=np.array([1.0, 0.0, 0.0]),
        show_progress=True
    )

    # Verify both methods agree
    agreement = np.sum(inside_mask == inside_mask_alt)
    print(f"\nMethod agreement: {agreement}/{n_test} ({agreement/n_test*100:.2f}%)")
    """

    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"Points inside: {np.sum(inside_mask):,}/{n_test:,} ({np.sum(inside_mask)/n_test*100:.1f}%)")
    print(f"Points outside: {n_test - np.sum(inside_mask):,}/{n_test:,}")

    # Save to .npz file
    output_file = 'test_results_trimesh.npz'
    np.savez_compressed(output_file,
         test_points=test_points,
         inside_mask=inside_mask)
    print(f"\nResults saved to: {output_file}")

    # Print sample test points
    print(f"\n{'='*60}")
    print("SAMPLE TEST POINTS")
    print('='*60)
    print("\nFirst 5 test points:")
    for i in range(min(5, n_test)):
        status = "INSIDE" if inside_mask[i] else "OUTSIDE"
        #print(f"  Point {i}: phi1={test_points[i,0]:7.2f}, phi2={test_points[i,1]:7.2f}, "
              #f"d={test_points[i,2]:.4f} → {status}")
        print(f"  Point {i}: phi1={test_points[i,0]:7.2f}, phi2={test_points[i,1]:7.2f}, "
              f"d={test_points[i,2]/400:.4f} → {status}")



    code_end = time.time()
    total_time = code_end - code_start
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print('='*60)
    print(f"  Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Mesh creation: {stats['mesh_creation_time']:.2f}s")
    print(f"  Ray casting: {stats['ray_casting_time']:.2f}s")
    #print(f"  Processing rate: {stats['rate']:.1f} points/second")
