import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_bifurcation():
    """3D plot with d, phi1, phi2 as axes"""

    # CORRECT way to load NPZ file
    # Option 1: Using context manager (automatically closes)
    with np.load('failed_points.npz') as f:
        phi1 = f['failed_points'][:,0]
        phi2 = f['failed_points'][:,1]
        d = f['failed_points'][:,2]
        #d = f['d'][:]
        #phi1 = f['phi1'][:]
        #phi2 = f['phi2'][:]
        #par5 = f['par5'][:]
        #par6 = f['par6'][:]
    # File is automatically closed after exiting the 'with' block
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot boundary points
    scatter = ax.scatter(phi1, phi2, d,
                        #c=par5,  # Color by par5 or par6
                        #cmap='jet',
                        s=20,
                        label='Boundary points',
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.3)
    
    # Colorbar
    #cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    #cbar.set_label('par(5)', fontsize=12)
    
    # Labels
    ax.set_xlabel('phi1', fontsize=12)
    ax.set_ylabel('phi2', fontsize=12)
    ax.set_zlabel('d', fontsize=12)
    ax.set_title('3D Bifurcation Diagram - Boundary Points', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.view_init(elev=0, azim=0)
    
    plt.tight_layout()
    plt.savefig('bifurcation_3d_boundary.png', dpi=150)
    plt.show()

# Usage
plot_3d_bifurcation()
