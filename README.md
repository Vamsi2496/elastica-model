# elastica-model
The elastica clamped at both the ends is modeled here. The input for the model is clamp angles at left end, right end and ratio of distance between ends and length of elastica. the outputs are force components, moment at let end and right end, energy stored in the elastica.


This repository consists of code to generate required number of layered data to get the initial guess and the boundary points of snapping. The data is generated using AUTO 07P: CONTINUATION AND BIFURCATION SOFTWARE FOR ORDINARY DIFFERENTIAL EQUATIONS

From the boundary point data, surface mesh is created using the Advancing front surface reconstruction algorithm available in CGAL library. 

This mesh helps to generte a random point and distinguish it between inside or outside of the domain. The ray casting method in trimesh library is used for the filtering.

With the generated data of initial set of points, Neural Network model is trained to predict the force components, moments and profile (i.e., arc length and corresponding angle)
