#!/usr/bin/env python3
"""
P-Median Problem Metaheuristic Solver

This script solves p-median problem instances using the gb21_mh metaheuristic algorithm.
It processes all .txt and .dat files in a specified directory and saves solutions
to an output directory with a '_mh_sol' suffix.

Usage:
    python pmedian_mh_solver.py --dir input_directory [--batch_size batch_size]
"""

import os
import sys
import numpy as np
import argparse
import time
import csv
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Import the metaheuristic algorithm
from algorithm import gb21_mh, read_inst

def solve_pmedian_mh(instance_path, time_limit=3600, threads=5):
    """Solve the P-Median Problem using the gb21_mh metaheuristic with fixed parameters."""
    start_time = time.time()
    
    # Fixed parameters as per example
    n_start = 1
    g_initial = 10
    init = "kmeans++"
    n_target = 200
    l = 50
    t_local = 300
    mip_gap_global = 0.01
    mip_gap_local = 0.01
    np_seed = 4
    gurobi_seed = 4
    no_local = False
    no_solver = False
    
    # Read problem instance
    X, Q, q, p = read_inst(instance_path)
    n = len(X)
    
    # Adjust parameters for large instances
    if n >= 5000:
        init = "capacity-based"
        l = 10
    
    # Apply the metaheuristic algorithm
    medians, assignments, algorithm_time, best_ofv = gb21_mh(
        X, Q, q, p, time_limit,
        n_start, g_initial, init,
        n_target, l, t_local,
        mip_gap_global, mip_gap_local,
        np_seed, gurobi_seed,
        no_local, no_solver, threads
    )
    
    total_time = time.time() - start_time
    
    return n, p, best_ofv, medians, assignments, total_time, algorithm_time

def save_solution(filename, n, p, obj_val, medians, assignments, runtime, algorithm_time):
    """Save solution to a file."""
    with open(filename, 'w') as f:
        # Write problem size and objective value
        f.write(f"{n} {p} {obj_val}\n")
        
        # Write runtime information
        f.write(f"# Total Runtime: {runtime:.2f} seconds\n")
        f.write(f"# Algorithm Runtime: {algorithm_time:.2f} seconds\n")
        
        # Write selected medians
        median_status = np.zeros(n, dtype=int)
        for i in medians:
            median_status[i] = 1
        f.write(' '.join(map(str, median_status)) + '\n')
        
        # Write assignments
        for j in range(n):
            f.write(f"{assignments[j]}\n")

def process_instance(args):
    """Process a single instance (for parallel execution)"""
    filename, instance_path, output_path, time_limit, threads = args
    
    print(f"Solving instance: {filename}")
    
    try:
        # Apply metaheuristic solver
        n, p, obj_val, medians, assignments, runtime, algorithm_time = solve_pmedian_mh(
            instance_path, time_limit, threads
        )
        
        # Save solution
        save_solution(output_path, n, p, obj_val, medians, assignments, runtime, algorithm_time)
        
        # Format values for summary
        obj_val_str = f"{obj_val:.6f}" if obj_val is not None else "N/A"
        
        # Add to summary file
        summary_file = os.path.join(os.path.dirname(output_path), 'summary_results.csv')
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,             # Instance
                n,                    # Points
                p,                    # Medians
                obj_val_str,          # Objective
                "N/A",                # DualBound (not available in metaheuristic)
                obj_val_str,          # PrimalBound 
                f"{algorithm_time:.2f}",     # Runtime
                "Completed"           # Status
            ])
        
        print(f"  Points: {n}, Medians: {p}")
        print(f"  Objective: {obj_val_str}")
        print(f"  Runtime: {algorithm_time:.2f} seconds")
        print(f"  Result added to summary file")
        
        return True
    
    except Exception as e:
        print(f"  Error solving instance: {str(e)}")
        
        # Add error entry to summary file
        summary_file = os.path.join(os.path.dirname(output_path), 'raw_results.csv')
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,  # Instance
                "Error",   # Points
                "Error",   # Medians
                "Error",   # Objective
                "N/A",     # DualBound
                "N/A",     # PrimalBound
                "0.0",     # Runtime
                "Error"    # Status
            ])
        
        print(f"  Error result added to summary file")
        return False

def main():
    parser = argparse.ArgumentParser(description='Solve P-Median Problem instances using Metaheuristic')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing problem instances')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='Time limit in seconds (default: 3600)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of instances to solve in parallel (default: 1)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads to use (default: 0, uses all available)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.dir + "_sol"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    # Create summary file with header
    summary_file = os.path.join(output_dir, 'summary_results.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance", "Points", "Medians", "Objective", "DualBound", 
            "PrimalBound", "Runtime", "Status"
        ])
    
    # Get list of instance files to solve (.txt and .dat files)
    instance_files = sorted([f for f in os.listdir(args.dir) 
                      if (f.endswith('.txt') or f.endswith('.dat'))])
    
    print(f"Found {len(instance_files)} instances to solve")
    
    # Process instances in batches
    available_cores = multiprocessing.cpu_count()
    effective_batch_size = min(args.batch_size, available_cores, len(instance_files))
    
    threads_per_instance = max(1, int(available_cores / effective_batch_size)) if args.threads == 0 else max(1, int(args.threads / effective_batch_size))
    # Prepare batch arguments
    batch_args = []
    for filename in instance_files:
        instance_path = os.path.join(args.dir, filename)
        output_path = os.path.join(output_dir, filename)
        batch_args.append((filename, instance_path, output_path, args.time_limit, threads_per_instance))
    
    if effective_batch_size > 1:
        print(f"Using batch processing with {effective_batch_size} parallel instances")
        # Process batches
        with ProcessPoolExecutor(max_workers=effective_batch_size) as executor:
            executor.map(process_instance, batch_args)
    else:
        # Sequential processing
        for arg in batch_args:
            process_instance(arg)
    
    print(f"Summary of results saved to {summary_file}")

if __name__ == "__main__":
    main()