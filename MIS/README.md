# Maximum Independent Set

## Overview
This repository provides the tools for generating and solving the MIS instances. 

- solve.sh — run KaMIS on a directory of graph instances in parallel  
- generate_training_instances.py — generate random RB MIS instances in DIMACS format  
- transform_kamis.py — convert DIMACS graph files into the METIS format expected by KaMIS
- summary.py — aggregate solver logs into a summary file
- config.py  — evaluation pipeline for LLM agents on MIS 

## Requirements
- NetworkX  
- NumPy  
- SciPy  
- tqdm  
- KaMIS (download and install manually; see below)  

## KaMIS Setup
1. Visit (https://github.com/KaMIS/KaMIS) 
2. Download the latest release for your platform  
3. Extract and place the redumis executable into this directory, or update the --executable path in solve.sh  

## Generate Training Instances

    python generate_training_instances.py \
      --num_graph 100 \
      --graph_type large \
      --seed 42 \
      --save_dir valid_instances \
      --output_prefix RB

### Solve with KaMIS


    python transform_kamis.py valid_instances valid_instances_kamis
    
    bash solve.sh \
      --instance-dir valid_instances \
      --parallel 8 \
      --executable ./redumis \
      --time-limit 3600


Options for solve.sh:
* `-i, --instance-dir DIR` -   directory containing METIS files (default: easy_test_instances)
* `-p, --parallel NUM` -      max parallel processes (default: 10)
* `-e, --executable PATH` -    path to redumis binary (default: ./redumis)
* `-t, --time-limit SEC` -     time limit per instance (default: 3600)
* `-h, --help` -               show this help message

Results are saved to `valid_instances_sol/`. 

## Summarize Results
    python summary.py \
      valid_instances \
      valid_instances_sol/summary_results.csv

This reads all `*.log` files in `valid_instances_sol/` and writes the aggregated CSV.





