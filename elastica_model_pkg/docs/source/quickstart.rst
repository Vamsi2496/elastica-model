Quick Start
===========

Basic usage
-----------

Data generation for a layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_generation_only_boundary() renders data for boundary points only .

run_generation() renders data for inplane points along with boundary points.

.. code-block:: python

   from elastica_model import run_generation,  run_generation_only_boundary

   total, phi1_values, phi2_values, d_values, hdf5_indices= run_generation_only_boundary(
       0.9997,
       hdf5_file="data.h5", 
       rtree_prefix="index",
       keep_AUTO_folders=False
   )  
   total, phi1_values, phi2_values, d_values, hdf5_indices= run_generation(
       0.9997,
       hdf5_file="data.h5", 
       rtree_prefix="index",
       keep_AUTO_folders=False
   )  
   

**Output variables:**

* ``total``        — number of solution blocks found
* ``phi1_values``  — boundary angle at the left end for each block
* ``phi2_values``  — boundary angle at the right end for each block
* ``d_values``     — clamp-to-clamp distance for each block
* ``hdf5_indices`` — list of row indices for direct HDF5 access

.. note::

   Two additional files are created in the working directory for
   reference and further querying:

   * ``data.h5``       — HDF5 file containing all solution data
   * ``index.dat / index.idx`` — R-tree spatial index for fast lookup

Generating data for a Specific Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a database has been built with :func:`run_generation()` or :func:`run_generation_only_boundary()` , use
:func:`run_at_point` to compute the solution at any specific
:math:`(\phi_1, \phi_2, d)` target. The function automatically finds the
nearest existing solution in the database, uses it as an initial condition
for AUTO, and appends the new result to the same HDF5 file and R-tree index. Make sure that, the target point is within the snapping envelope. 
Otherwise convergence is not guaranteed.

.. code-block:: python

   from elastica_model import run_at_point

   convergence, phi1_values, phi2_values, d_values, hdf5_indices = run_at_point(
       phi1         = 0.45,
       phi2         = -0.30,
       d            = 0.85,
       hdf5_file    = "data.h5",
       rtree_prefix = "index"
   )



**Output variables:**

* ``convergence``  — ``True`` if AUTO successfully converged
* ``phi1_values``  — boundary angle at the left clamp for each new block
* ``phi2_values``  — boundary angle at the right clamp for each new block
* ``d_values``     — clamp-to-clamp distance for each new block
* ``hdf5_indices`` — direct row indices of the newly appended blocks

.. note::

   The database grows incrementally with every successful call.

.. warning::

   Calling ``run_at_point()`` on an empty database will raise a
   ``FileNotFoundError``.

HDF5 File Structure
-------------------

The HDF5 file stores the following datasets, one row per solution block:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Dataset
     - Description
   * - ``d``
     - Clamp-to-clamp distance
   * - ``phi1``
     - Boundary angle at the left end
   * - ``phi2``
     - Boundary angle at the right end
   * - ``t``
     - arc length at mesh nodes
   * - ``u1``
     - tangent angle at mesh nodes
   * - ``inflection_points``
     - number of curvature changes in the beam
   * - ``parameters``
     - Array of 8 load/geometry values per block (see table below)

The ``parameters`` array has the following layout:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Index
     - Quantity
   * - ``parameters[0]``
     - :math:`F_x` — horizontal reaction force
   * - ``parameters[1]``
     - :math:`F_y` — vertical reaction force
   * - ``parameters[2]``
     - :math:`x_\text{tip}` — tip x-coordinate
   * - ``parameters[3]``
     - :math:`y_\text{tip}` — tip y-coordinate
   * - ``parameters[4]``
     - :math:`\frac{phi1 + phi2}{2}` — Asymmetric angle
   * - ``parameters[5]``
     - :math:`\frac{phi2 - phi1}{2}` — Symmetric angle
   * - ``parameters[6]``
     - Moment at left end
   * - ``parameters[7]``
     - Moment at right end
   * - ``parameters[8]``
     - Energy stored in the beam

Access results from HDF5
------------------------

Use the returned ``hdf5_indices`` to read directly:

.. code-block:: python

   import h5py
   idx  = hdf5_indices[0]
   with h5py.File("auto_data.h5", "r") as f:
       d          = f["d"][idx]
       phi1       = f["phi1"][idx]
       phi2       = f["phi2"][idx]
       s          = f["t"][idx]
       theta      = f["u1"][idx]
       params     = f["parameters"][idx]   # shape (9,)
       Fx         = params[0]
       Fy         = params[1]
       x_tip      = params[2]
       y_tip      = params[3]
       Asymmetric = params[4]
       Symmetric  = params[5]
       M1         = params[6]       # Moment at left end
       M2         = params[7]       # Moment at right end
       PE         = params[8]
       inflection_points       = f["inflection_points"][idx]

.. note::

   ``hdf5_indices[i]`` maps directly to the HDF5 row —
   no search needed.

.. warning::

   d = 1.0 is degenerate and it is recommended to use less than 0.9999
