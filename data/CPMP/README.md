# Capacitated P-Median Problem

## Overview
This directory provides tools for generating and solving capacitated p-median (CPMP) instances.

- generate_training_instances.py — generate p-median instances 
- solve.py                     — batch/sequential metaheuristic solver (gb21_mh)  
- algorithm.py                 — implements `gb21_mh` and `read_inst`
- config.py                    — evaluation pipeline for LLM agents on CPMP 

## Requirements
- NumPy
- SciPy
- Scikit Learn
- tqdm
- Gurobi Python interface (gurobipy)  
- Gurobi license 

## Generate Training Instances
    python generate_training_instances.py \
      --n 500 \
      --p 5 10 20 50 \
      --instances 5 \
      --seed 42 \
      --output_dir valid_instances

This writes `.txt` files into `valid_instances/`.

## Solve CPMP Instances
    python solve.py \
      --dir valid_instances \
      --time_limit 3600 \
      --batch_size 10 \
      --threads 10

Options for `solve.py`:
* `--dir DIR`            — input directory containing `.txt` instances  
* `--time_limit SEC`     — time limit per instance (default: 3600)  
* `--batch_size N`       — parallel instances (default: 1)  
* `--threads N`          — threads per instance (0 = all available; default: 0)  

Results are saved to `valid_instances_sol/` with individual solution files and `summary_results.csv`.  