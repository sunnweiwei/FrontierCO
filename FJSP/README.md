# Flexible Job Shop Problem (FJSP)

## Overview
This directory provides tools for generating and solving FJSP instances.

- generate_training_instances.py — generate FJSP `.fjs` (and/or `.pkl`) instances (`valid_instances/`)  
- transform_to_CP.py              — convert `.fjs` to CP text format (`valid_instances_cp/`)  
- solve.py                        — multi‐problem solver entry point (CP/CPL or Google)
- baselines                       - library of various classical solvers
- summary.py                      — aggregate FJSP results into CSV
- config.py                       — evaluation pipeline for LLM agents on CFLP 

## Requirements
- NumPy  
- [CPLEX](https://www.ibm.com/docs/en/icos)
- OR-Tools (Optional)
- Gurobi (Optional)
- tqdm  

## Generate FJSP Instances
    python generate_training_instances.py \
        --instance_type standard \
        --n_jobs 20 \
        --n_machines 10 \
        --op_per_job 30 \
        --data_dir valid_instances \
        --n_data 20

This writes `.fjs` files into `valid_instances/`.

## Convert to CP Format
    python transform_to_CP.py \
      valid_instances \
      valid_instances_cp

This reads `valid_instances/*.fjs` and writes CP‐style files into `valid_instances_cp/`.

## Solve FJSP Instances
    python solve.py \
      --problem Flexiblejobshop \
      --model CP \
      --time 60 \
      --solver-name CPLEX \
      --threads 10 \
      --input valid_instances

Options for `solve.py`:
* `--problem`       — problem type (`Flexiblejobshop`)  
* `--model`         — model type (`CP`)  
* `--time`          — time limit in seconds  
* `--solver-name`   — `CPLEX`  
* `--threads`       — threads per instance  
* `--input`         — input directory (`valid_instances/`)  

Results are saved to `valid_instances_sol/`.

## Summarize Results
    python summary.py \
      valid_instances_sol \
      valid_instances_sol/summary_results.csv

This reads `valid_instances_sol/result*` files and writes `valid_instances_sol/summary_results.csv`. 