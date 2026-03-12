# About elastica-model
The elastica clamped at both the ends is modeled here. The input for the model is clamp angles at left end (phi1), right end (phi2)and Ratio of clamp-to-clamp distance to arch length (d/L). The outputs are force components, moment at let end and right end, energy stored in the elastica.

This repository consists of code to generate required number of layered data to get the initial guess and the boundary points of snapping. The data is generated using AUTO 07P: CONTINUATION AND BIFURCATION SOFTWARE FOR ORDINARY DIFFERENTIAL EQUATIONS

From the boundary point data, surface mesh is created using the Advancing front surface reconstruction algorithm available in CGAL library. This mesh helps to generte a random point and distinguish it between inside or outside of the domain. The ray casting method in trimesh library is used for the filtering.

Using the mesh and initial data, new data can be generated and is appended to the .h5 file and rtree is updated.

##Prerequisites
The following softwares need to be installed prior to using this.
1. **AUTO-07p**
2. **CGAL Library**

**Input required:**
For initial data generation:
The range of layers and their spacing. update it in "initial_data_generation/initial_data_generation.auto"
Default is \( 0.6 :0.01: 0.99 \)

Mesh generation:
Update the above layer and spacing information in the file "boundary_only.py"

For data generation using initial guess:
the number of data points user want to generate data for and the number of CPU to be used

All calculations are nondimensionalized such that:

- Length of the Elastica = 1  
- Bending modulus **EI = 1**  


##How to run
1. Open the AUTO CLUI. change the directory to the current folder. run the script using: auto.auto('initial_data_generation.auto')
2. Once the data generation is over, run the python programme "automated_parsing.py". This will create the .h5 file and rtree index files
3. Cut and paste the .h5 and rtree files into Mesh generation folder. run the python programme "boundary_only.py".It will create a .xyz file with boundary points.
4. run the CGAL codes to generate the mesh.
5. Cut and paste the .h5, rtree files and the mesh file into "Data Generation" folder. 
6. update the directory  to the "Data Generation" folder in "master_loop_parallel.py" file.
7. Run the "master_loop_parallel.py" code

## Results

The .h5 file consists of phi1, phi2, d, Fx, Fy, Moment at left clamp and Moment at right clamp, arc length and corresponding theta and number of inflection points of the profile

## 📎 Additional Files
For details, user can refer to [AUTO-07p documentation](https://depts.washington.edu/bdecon/workshop2012/auto-tutorial/documentation/auto07p%20manual.pdf).