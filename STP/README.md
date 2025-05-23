# Steiner Tree Problem

## Overview
This directory provides tools for generating and solving Steiner Tree Problem (STP) instances.

- generate_training_instances.py — generate STP `.stp` instances
- solve.sh                       — solve STP instances via SCIP-Jack 
- write.set                      - Solver parameters
- config.py                      — evaluation pipeline for LLM agents on STP

## Requirements
- NumPy  
- NetworkX  
- tqdm  
- SCIP-Jack (download and install manually; see below)  

## SCIP-Jack Setup

Please refer to [SCIP-Jack](https://scipjack.zib.de/) for the installation instruction. We obtain the binary file of SCIP-Jack from [Daniel Rehfeldt](mailto:rehfeldt@zib.de). Put the binary file under this directory and specify it during the solving time.

## Generate Training Instances
Here is an example to generate the hypercube instances. In `generate_training_instances.py`, we also include some other generators corresponding to the instances in [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/steininfo.html). But in general, these instances are easy to solve unless for large and dense ones. Please refer to the our curated [training set](https://huggingface.co/datasets/CO-Bench/FrontierCO-Train/tree/main/STP).

    python generate_training_instances.py \
      --type hypercube \
      --dimensions 6 7 8 9 10 \
      --instances_per_config 2 \
      --seed 42 \
      --output_dir valid_instances

This writes `.stp` files into `valid_instances/`.


## Solve STP Instances
    bash solve.sh valid_instances /path/to/stp.linux.x86_64.gnu.opt.cpx write.set 8

Arguments for `solve.sh`:
* `<input_folder>`       — directory containing `.stp` files (e.g. `valid_instances`)  
* `<solver_binary>`      — path to the SteinLib solver binary (e.g. `/usr/local/bin/stp.linux.x86_64.gnu.opt.cpx`)  
* `<param_file>`         — solver parameter file (e.g. `write.set`)  
* `[max_parallel_jobs]`  — optional max parallel jobs (default: number of CPU cores)  

You can specify the solver parameters in write.set, which are basically the same as the ones in [SCIP](https://www.scipopt.org/).

Results and `.stplog` files are moved into `valid_instances_sol/`.  
