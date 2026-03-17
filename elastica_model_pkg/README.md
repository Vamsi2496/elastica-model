# Elastica Model

Elastica bifurcation model — parallel AUTO data generation and parsing.

## Install
Run inside the package root (where pyproject.toml lives):  

example  
cd C:\Users\sanch\Desktop\elastica_model_pkg

python pip install -e .  
(note the '.' at the end of the line. it is required)


## First-time setup (set your Python 2.6 and AUTO paths)  
Run from anywhere — it just saves paths to ~/.elastica_model/config.json:  

elastica-config

## Run data generation
Run from the folder where you want the data to be generated — this is where auto_data.h5, auto_rtree_index.*,
and d0p*/ folders will be created:  

elastica-generate

## Use as a library
*from elastica_model import run_generation, parse_folders*  
succeeded, failed, folders = run_generation(0.60, 0.99, 0.01, n_workers=4)  
parse_folders(folders, hdf5_file="auto_data.h5")
