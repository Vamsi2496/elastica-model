import numpy as np
import time
from pathlib import Path
import trimesh
#import pyembree

# ==================== TRIMESH RAY CASTING ====================

def test_points_trimesh(points, vertices, faces, show_progress=True):
    """
    Test multiple points using trimesh's built-in ray casting IN BATCHES.

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
    BATCH_SIZE = 1000  # Fixed batch size for memory control

    print(f"\n  Creating trimesh object...")
    t_mesh_start = time.time()

    # Create trimesh object ONCE
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #mesh = trimesh.Trimesh(vertices=vertices, faces=faces, use_embree=True)

    t_mesh_end = time.time()
    print(f"  ✓ Mesh created in {t_mesh_end - t_mesh_start:.2f}s")
    print(f"  Mesh info: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print(f"  Watertight: {mesh.is_watertight}, Volume: {mesh.volume:.4f}")

    print(f"\n  Testing {n_points:,} points with trimesh ray casting (batches of {BATCH_SIZE:,})...")
    t_start = time.time()

# added to make code faster
    mesh.use_embree = True
    #print("Embree:", hasattr(mesh.ray, 'query_pyembree'))
    
    # Pre-allocate output
    inside_mask = np.empty(n_points, dtype=bool)
    
    # Process in batches
    n_batches = (n_points + BATCH_SIZE - 1) // BATCH_SIZE
    batch_times = []
    
    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, n_points)
        batch_points = points[start_idx:end_idx]
        
        batch_start = time.time()
        inside_mask[start_idx:end_idx] = mesh.contains(batch_points)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if show_progress:
            progress = (i + 1) / n_batches * 100
            print(f"    Batch {i+1:4d}/{n_batches:4d} ({progress:6.1f}%) - {len(batch_points):6,} pts in {batch_time:.3f}s")
    
    t_total = time.time() - t_start

    stats = {
        'total_time': t_total,
        'mesh_creation_time': t_mesh_end - t_mesh_start,
        'ray_casting_time': t_total,
        'batch_times': batch_times,
        'avg_batch_time': np.mean(batch_times),
        'rate': n_points / t_total if t_total > 0 else 0,
        'total_faces': len(faces),
        'is_watertight': mesh.is_watertight,
        'n_batches': n_batches
    }

    print(f"\n  ✓ Completed in {t_total:.2f}s ({stats['rate']:.1f} points/sec)")
    print(f"  Average batch time: {stats['avg_batch_time']:.3f}s")

    return inside_mask, stats


# ==================== USAGE ====================

if __name__ == "__main__":

    print("="*60)
    print("TRIMESH RAY CASTING TEST (MILLIONS OF POINTS)")
    print("="*60)

    # Load surface mesh
    print("\nLoading surface mesh...")
    CACHE_DIR = Path('delaunay_cache')
    
    #with np.load(CACHE_DIR / 'boundary_3d.npz') as f:
     #   boundary_3d = f['boundary_3d']
    #with np.load(CACHE_DIR / 'delaunay_surface.npz') as f:
     #   delaunay_surface_faces = f['surface_faces']
        
    mesh = trimesh.load("final_mesh.off", force='mesh')
    boundary_3d = mesh.vertices
    delaunay_surface_faces = mesh.faces

    print(f"  Vertices: {len(boundary_3d):,}")
    print(f"  Faces: {len(delaunay_surface_faces):,}")

    # Generate test points
    phi1_min, phi1_max = boundary_3d[:, 0].min(), boundary_3d[:, 0].max()
    phi2_min, phi2_max = boundary_3d[:, 1].min(), boundary_3d[:, 1].max()
    d_min, d_max = boundary_3d[:, 2].min(), boundary_3d[:, 2].max()

    #d_min = 0.60*50
    #d_max = 0.99*50

    print(f"\nBounding box:")
    print(f"  phi1: [{phi1_min:.3f}, {phi1_max:.3f}]")
    print(f"  phi2: [{phi2_min:.3f}, {phi2_max:.3f}]")
    print(f"  d:    [{d_min:.4f}, {d_max:.4f}]")

    code_start = time.time() 

    n_test = 1000
    print(f"\nGenerating {n_test:,} random test points...")
    #test_points = np.random.uniform([phi1_min, phi2_min, d_min],[phi1_max, phi2_max, d_max],size=(n_test, 3))
    test_points = np.random.uniform([0, 0, d_min],[0, 0, d_max],size=(n_test, 3))
    
    print(f"  ✓ Generated {len(test_points):,} points ({test_points.nbytes/1e9:.1f} GB)")

    # METHOD 1: Use trimesh's built-in contains method with batching (RECOMMENDED)
    print("\n" + "="*60)
    print("METHOD 1: trimesh.contains() BATCHED [RECOMMENDED FOR MILLIONS]")
    print("="*60)
    inside_mask, stats = test_points_trimesh(
        test_points, boundary_3d, delaunay_surface_faces,
        show_progress=True
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    n_inside = np.sum(inside_mask)
    print(f"Points inside:  {n_inside:,}/{n_test:,} ({n_inside/n_test*100:.1f}%)")
    print(f"Points outside: {n_test - n_inside:,}/{n_test:,}")
    inside_points = test_points[inside_mask]
    # Save to .npz file (compressed)
    output_file = 'test_results_trimesh_inside_only.npz'
    print(f"\nSaving results ({inside_mask.nbytes/1e6:.1f} MB)...")
    np.savez_compressed(output_file,
         #test_points=test_points,
         inside_points=inside_points,
         inside_mask=inside_mask)
    print(f"  ✓ Results saved to: {output_file}")

    # Print sample test points
    print(f"\n{'='*60}")
    print("SAMPLE TEST POINTS")
    print('='*60)
    print("\nFirst 5 test points:")
    #for i in range(min(5, n_test)):
    for i in range( n_test):
        status = "INSIDE" if inside_mask[i] else "OUTSIDE"
        print(f"  Point {i}: phi1={test_points[i,0]:7.2f}, phi2={test_points[i,1]:7.2f}, "
              f"d={test_points[i,2]:.4f} → {status}")
    code_end = time.time()
    total_time = code_end - code_start
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print('='*60)
    print(f"  Total runtime:     {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Mesh creation:     {stats['mesh_creation_time']:.2f}s")
    print(f"  Ray casting:       {stats['ray_casting_time']:.2f}s")
    print(f"  Batches processed: {stats['n_batches']:,}")
    print(f"  Processing rate:   {stats['rate']:.1f} points/second")
    #print(f"  Memory (points):   {test_points.nbytes/1e9:.1f} GB")
