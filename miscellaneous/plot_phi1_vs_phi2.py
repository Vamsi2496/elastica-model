import numpy as np
import h5py
from rtree import index
import matplotlib.pyplot as plt

def plot_bifurcation_at_d_styled(d_target, tolerance=0.001, non_blocking=False):
    """
    Plot phi1 vs phi2 at specific d value.
    Color by PAR(9), marker by inflection points.
    """
    # Load R-tree index
    p = index.Property()
    p.dimension = 3
    idx = index.Index('auto_rtree_index', properties=p)
    
    try:  # Use try-finally to ensure index is closed
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
        
    finally:
        # CRITICAL: Always close the index
        idx.close()
        print("R-tree index closed")
    
    print(f"Found {len(match_indices)} points at d = {d_target}")
    
    if len(match_indices) == 0:
        print("No points found!")
        return
    match_indices = sorted(match_indices)
    # Load data from HDF5
    with h5py.File("auto_data.h5", 'r') as f:
        phi1 = f['phi1'][match_indices]
        phi2 = f['phi2'][match_indices]
        par = f['parameters'][match_indices]
        inflection_points = f['inflection_points'][match_indices]
    
    # Extract PAR(9) - the 9th parameter (index 8)
    par9 = par[:, 8]
    
    # Define markers for different inflection point values
    #markers = {0: 's', 1: '^', 2: 'o', 'other': 'x'}
    #marker_sizes = {0: 20, 1: 20, 2: 5, 'other': 20}
    markers = {0: 's', 1: '^', 2: 'o', 3: 'D', 'other': 'x'}
    marker_sizes = {0: 20, 1: 20, 2: 5, 3: 20, 'other': 20}
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each inflection point group separately
    scatter = None
    for sign_change_value in [0, 1, 2, 3]:
        mask = (inflection_points == sign_change_value)
        if np.any(mask):
            scatter = ax.scatter(phi1[mask], phi2[mask], 
                                c=par9[mask], 
                                marker=markers[sign_change_value],
                                cmap='jet',  
                                s=marker_sizes[sign_change_value],
                                label=f'Inflection points={sign_change_value}',
                                alpha=0.7,
                                edgecolors='black',
                                linewidths=0.5,
                                vmin=par9.min(),  # Ensure consistent colormap
                                vmax=par9.max())
    
    # Plot all other inflection point values with a different marker
    other_mask = (inflection_points > 3) | (inflection_points < 0)
    if np.any(other_mask):
        scatter = ax.scatter(phi1[other_mask], phi2[other_mask],
                            c=par9[other_mask],
                            marker=markers['other'],
                            cmap='jet',  
                            s=marker_sizes['other'],
                            label='inflection points=other',
                            alpha=0.7,
                            #edgecolors='black',
                            linewidths=0.5,
                            vmin=par9.min(),
                            vmax=par9.max())
    
    # Add colorbar
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Energy', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('phi1', fontsize=12)
    ax.set_ylabel('phi2', fontsize=12)
    ax.set_title(f'd={d_target:.2f}', fontsize=14)
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'phi1_vs_phi2_d_{d_target:.2f}.png', dpi=150)
    # Control blocking behavior
    if non_blocking:
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)  # Non-blocking - terminal available immediately
        print("Plot displayed (non-blocking). Close manually when done.")
    else:
        plt.show()  # Blocking - waits for window to close
    
    print(f"Plot saved as 'phi1_vs_phi2_d_{d_target:.2f}.png'")

    

# Example usage
if __name__ == "__main__":
    # Option 1: Plot specific d value
    plot_bifurcation_at_d_styled(d_target=0.59, tolerance=0.001)
    
    # Option 2: Plot all d values
    # plot_all_d_values_styled()
