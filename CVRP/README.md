# Capacitated Vehicle Routing Problem

## Overview
This directory provides tools for generating and solving CVRP instances.

- generate_training_instances.py — generate CVRP `.vrp` instances 
- cvrp_solver_hgs.py           — solve CVRP via HGS (PyHgese)  
- config.py                     — evaluation pipeline for LLM agents on CVRP  

## Requirements 
- NumPy
- tqdm  
- (PyHgese)[https://github.com/chkwon/PyHygese] 

## Generate Training Instances
    python generate_training_instances.py \
      --sizes 20 50 100 \
      --instances 5 \
      --capacity 50 \
      --demand_low 1 \
      --demand_high 9 \
      --depot_x 0.5 \
      --depot_y 0.5 \
      --seed 42 \
      --output valid_instances

This writes `.vrp` files into `valid_instances/`.

## Solve CVRP Instances
    python cvrp_solver_hgs.py \
      --dir valid_instances \
      --output valid_instances_sol \
      --time_limit 3600.0 \
      --threads 10 \
      --seed 1 \
      --batch_size 10

Options for `cvrp_solver_hgs.py`:
* `--dir DIR`           — input directory containing `.vrp` instances  
* `--output DIR`        — output directory for solutions (default: `<dir>_sol`)  
* `--time_limit SEC`    — time limit in seconds (default: 3600.0)  
* `--threads N`         — threads per instance (0 = all available; default: 0)  
* `--seed N`            — random seed for reproducibility (default: 1)  
* `--batch_size N`      — parallel instances (in our experiments: = # threads)  

Results are saved to `valid_instances_sol/` with individual solution files and `summary_results.csv`.