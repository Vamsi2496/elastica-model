# Elastica Model

Elastica model — parallel AUTO data generation and parsing.

## Install
Run inside the package root (where pyproject.toml lives):  

example  
cd C:\Users\sanch\Desktop\elastica_model_pkg

py -3.11 -m pip install -e .  
(note the '.' at the end of the line. it is required to make the installation editable)


## First-time setup (set your Python 2.6 and AUTO paths)  
Run from anywhere — it just saves paths to ~/.elastica_model/config.json:  

py -3.11 -m elastica_model.setup_config


## Use as a library
*from elastica_model import run_generation,  run_generation_only_boundary*  
for data of boundary points only, use the below line  
total, phi1_values, phi2_values, d_values, hdf5_indices= run_generation_only_boundary(0.9997, hdf5_file="data.h5", rtree_prefix="index",keep_AUTO_folders=False)  
for data of the entire layer, use the below line  
total, phi1_values, phi2_values, d_values, hdf5_indices= run_generation(0.99992, hdf5_file="data.h5", rtree_prefix="index", keep_AUTO_folders=False) 


## Run data generation
Run from the folder where you want the data to be generated — this is where auto_data.h5, auto_rtree_index.*,
and d0p*/ folders will be created:  

py -3.11 -m elastica_model.cli