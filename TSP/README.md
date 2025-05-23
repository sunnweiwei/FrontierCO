# Traveling Salesman Problem

## Overview
This directory provides tools for generating and solving TSP instances.

- generate_training_instances.py   — generate TSP `.tsp` instances  
- generate_par.sh                  — create LKH parameter files  
- solve.sh                         — solve TSP via LKH using `.par` files  
- summary.py                       — aggregate LKH logs into a CSV  
- config.py                        — evaluation pipeline for LLM agents on TSP

## Requirements
- NumPy    
- [LKH](http://webhotel4.ruc.dk/~keld/research/LKH-3/)

## Generate TSP Instances
    python generate_training_instances.py \
      --min_nodes 1000 \
      --max_nodes 1000 \
      --num_instances 20 \
      --seed 42 \
      --output valid_instances

This writes `.tsp` files into `valid_instances/`.

## Create LKH Parameter Files
    bash generate_par.sh valid_instances

This reads `valid_instances/*.tsp` and writes `.par` files into `valid_instances_par/`. Note that we select `POPMUSIC configuration` for our `hard_test_instances`.

## Solve TSP Instances
    bash solve.sh \
      --instance-dir valid_instances \
      --lkh-path /path/to/LKH \
      --parallel 10

Options for `solve.sh`:
* `-i, --instance-dir DIR` — intances directory (default: `easy_test_instances`)  
* `-l, --lkh-path PATH`    — path to LKH executable (default: `./LKH-3.0.13/LKH`)  
* `-p, --parallel NUM`     — max parallel jobs (default: 10)  

Results are saved to `valid_instances_sol/.  

## Summarize Results
    python summary.py \
      valid_instances \
      valid_instances_sol/summary_results.csv

This reads all `.log` files in `valid_instances_sol/` and writes the aggregated CSV.  
