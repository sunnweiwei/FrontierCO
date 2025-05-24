# Capacitated Facility Location Problem

## Overview
This directory provides tools for generating and solving CFLP instances.

- generate_training_instances.py — generate CFLP instances in text format  
- solve.py                     — Gurobi solver  
- algorithm.py                 — helper functions
- config.py                    — evaluation pipeline for LLM agents on CFLP  

## Requirements
- NumPy  
- tqdm  
- Gurobi

## Generate Training Instances

    python generate_training_instances.py \
      --customers 500 \
      --facilities 5 10 20 50 \
      --method cornuejols \
      --instances_per_config 5 \
      --seed 42 \
      --output_dir valid_instances

This writes `.txt` files into `valid_instances/`.

We also implemented the generator for the instances in [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html) with `--method beasley`. Feel free to explore this generator.

## Solve CFLP Instances
    python solve.py \
      --dir valid_instances \
      --time_limit 3600 \
      --threads 10 \
      --mip_gap 0 \
      --batch_size 10

Options for `solve.py`:
* `--dir DIR`            — input directory containing `.txt` instances  
* `--time_limit SEC`     — time limit per instance (default: 3600)  
* `--threads N`          — threads per instance (0 = all available; default: 0)  
* `--mip_gap FLOAT`      — relative MIP gap (default: 0)  
* `--batch_size N`       — parallel instances (default: 1)  

Results are saved to `valid_instances_sol/` with individual solution files and `summary_results.csv`.