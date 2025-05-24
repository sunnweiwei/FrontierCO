#!/usr/bin/env python3
"""
Minimum Dominating Set Problem Solver with Batch Processing

This script solves Minimum Dominating Set instances using Gurobi optimizer with batch processing capabilities.
It processes all input files in a specified directory and saves solutions
to an output directory with a '_sol' suffix.

Usage:
    python solve.py --dir input_directory [options]
"""

import os
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import argparse
import time
import csv
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def read_mds_problem(filename):
    """Read Minimum Dominating Set Problem instance from a file.
    
    Expected format:
    p ds N M  (N=vertices, M=edges)
    v1 v2  (edge between v1 and v2)
    v2 v3
    ...
    
    Note: Vertices are 1-indexed and the graph is undirected.
    Each edge appears only once in the file.
    """
    edges = []
    n = 0  # Number of vertices
    m = 0  # Number of edges
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Parse problem line (p ds n m)
            if parts[0] == 'p' and len(parts) >= 4 and parts[1] == 'ds':
                n = int(parts[2])
                m = int(parts[3])
            
            # Parse edge (v1 v2) - any line with two integers is considered an edge
            elif len(parts) >= 2:
                try:
                    v1 = int(parts[0])
                    v2 = int(parts[1])
                    edges.append((v1, v2))
                except ValueError:
                    # Skip lines that don't have two integers
                    continue
    
    if n == 0:
        raise ValueError("Invalid file format: Missing problem definition line (p ds n m)")
    
    # Create adjacency list representation of the graph (1-indexed)
    adjacency = {v: set() for v in range(1, n+1)}
    for v1, v2 in edges:
        if 1 <= v1 <= n and 1 <= v2 <= n:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)  # Undirected graph
        else:
            print(f"Warning: Edge ({v1}, {v2}) contains vertex outside range 1-{n}")
    
    # Print graph statistics
    print(f"  Graph with {n} vertices and {len(edges)} edges")
    
    return n, m, adjacency
def solve_mds(n, adjacency, time_limit, threads, mip_gap):
    """Solve the Minimum Dominating Set Problem using Gurobi.
    
    Parameters:
    n (int): Number of vertices
    adjacency (dict): Adjacency list representation of the graph
    time_limit (int): Time limit in seconds
    threads (int): Number of threads to use
    mip_gap (float): MIP gap for early termination
    """
    # Create model
    model = gp.Model("MDS")
    
    # Set parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)
    model.setParam('MIPGap', mip_gap)
    
    # Create variables
    # x[i] = 1 if vertex i is in the dominating set, 0 otherwise
    x = {}
    for i in range(1, n+1):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
    
    # Set objective function (minimize number of vertices in the dominating set)
    obj = gp.LinExpr()
    for i in range(1, n+1):
        obj += x[i]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Add constraint: each vertex must either be in the dominating set or adjacent to a vertex in the dominating set
    for i in range(1, n+1):
        expr = gp.LinExpr()
        # Add the vertex itself
        expr += x[i]
        # Add all neighbors
        for j in adjacency[i]:
            expr += x[j]
        
        model.addConstr(expr >= 1, f"domination_{i}")
    
    # Optimize model
    model.optimize()
    
    # Extract solution details
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Get objective value if available
        obj_val = model.ObjVal if model.SolCount > 0 else None
        
        # Extract solution if available
        if model.SolCount > 0:
            dominating_set = []
            for i in range(1, n+1):
                if x[i].x > 0.5:  # If vertex is in the dominating set
                    dominating_set.append(i)
            
            # Verify solution (for debugging)
            is_valid = verify_solution(dominating_set, adjacency, n)
            
            # Calculate dominance coverage
            coverage = calculate_coverage(dominating_set, adjacency, n)
        else:
            dominating_set = None
            is_valid = None
            coverage = None
        
        # Get dual bound (best lower bound)
        dual_bound = model.ObjBound if hasattr(model, 'ObjBound') else None
        
        # Get primal bound (best feasible solution)
        primal_bound = model.ObjVal if model.SolCount > 0 else None
        
        return obj_val, dominating_set, is_valid, coverage, model.Runtime, model.status, dual_bound, primal_bound, model.MIPGap if hasattr(model, 'MIPGap') else None
    else:
        return None, None, None, None, model.Runtime, model.status, None, None, None

def verify_solution(dominating_set, adjacency, n):
    """Verify that the dominating set is valid (all vertices are dominated)."""
    if not dominating_set:
        return False
    
    # Check that every vertex is either in the dominating set or adjacent to a vertex in it
    for i in range(1, n+1):
        if i in dominating_set:
            continue  # Vertex is in the dominating set
        
        # Check if vertex i is adjacent to any vertex in the dominating set
        if not any(j in dominating_set for j in adjacency[i]):
            return False  # Vertex i is not dominated
    
    return True

def calculate_coverage(dominating_set, adjacency, n):
    """Calculate coverage statistics for the dominating set."""
    # Count how many vertices are covered by each vertex in the dominating set
    coverage_count = {v: 0 for v in dominating_set}
    
    for i in range(1, n+1):
        # If vertex is in dominating set, it covers itself
        if i in dominating_set:
            coverage_count[i] += 1
        
        # For each neighbor in the dominating set, increment its coverage
        for j in adjacency[i]:
            if j in dominating_set:
                coverage_count[j] += 1
    
    return coverage_count

def save_solution(filename, obj_val, dominating_set, is_valid, coverage, runtime, n, m, status, dual_bound, primal_bound, gap):
    """Save solution to a file."""
    with open(filename, 'w') as f:
        # Write problem size and optimal objective value
        f.write(f"{n} {m} {int(obj_val) if obj_val is not None else 'N/A'}\n")
        
        # Write dominating set size and status
        if dominating_set is not None:
            f.write(f"{len(dominating_set)} {is_valid}\n")
            
            # Write vertices in the dominating set
            f.write(' '.join(map(str, sorted(dominating_set))) + '\n')
            
            # Write coverage information
            if coverage:
                for vertex, count in sorted(coverage.items()):
                    f.write(f"{vertex} {count}\n")
        else:
            f.write("0 False\n")
            f.write("N/A\n")
        
        # Write runtime and solution status as comments
        f.write(f"# Runtime (seconds): {runtime:.4f}\n")
        f.write(f"# Status: {status}\n")
        
        # Write bounds and gap information
        f.write(f"# Dual Bound: {dual_bound if dual_bound is not None else 'N/A'}\n")
        f.write(f"# Primal Bound: {primal_bound if primal_bound is not None else 'N/A'}\n")

def process_instance(args):
    """Process a single instance (for parallel execution)"""
    filename, instance_path, output_path, time_limit, threads, mip_gap, summary_file = args
    
    print(f"Solving instance: {filename}")
    
    try:
        # Read problem
        n, m, adjacency = read_mds_problem(instance_path)
        print(f"  Vertices: {n}, Edges: {m}")
        
        # Solve problem
        obj_val, dominating_set, is_valid, coverage, runtime, status, dual_bound, primal_bound, gap = solve_mds(
            n, adjacency, time_limit, threads, mip_gap
        )
        
        # Determine status string
        if status == GRB.OPTIMAL:
            status_str = "Optimal"
        elif status == GRB.TIME_LIMIT and obj_val is not None:
            status_str = "TIMEOUT"
        elif status == GRB.INFEASIBLE:
            status_str = "Infeasible"
        elif status == GRB.UNBOUNDED:
            status_str = "Unbounded"
        else:
            status_str = "Unknown"
        
        # Save solution
        save_solution(
            output_path, obj_val, dominating_set, is_valid, coverage, 
            runtime, n, m, status_str, dual_bound, primal_bound, gap
        )
        
        # Format values for summary
        if obj_val is not None:
            obj_val_str = f"{obj_val:.9f}"
            primal_bound_str = f"{primal_bound:.9f}"
        else:
            obj_val_str = "N/A"
            primal_bound_str = "N/A"
        
        if dual_bound is not None:
            dual_bound_str = f"{dual_bound:.9f}"
        else:
            dual_bound_str = "N/A"
        
        # Write result to summary file
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,              # Instance
                n,                     # Vertices
                m,                     # Edges
                obj_val_str,           # Objective (same as dominating set size)
                dual_bound_str,        # DualBound
                primal_bound_str,      # PrimalBound
                f"{runtime:.1f}",      # Runtime
                status_str             # Status
            ])
        
        print(f"  Status: {status_str}")
        print(f"  Objective: {obj_val_str}")
        if obj_val is not None:
            print(f"  Dominating Set Size: {len(dominating_set)}")
            print(f"  Solution Valid: {is_valid}")
        print(f"  Runtime: {runtime:.1f} seconds")
        if gap is not None:
            print(f"  Gap: {gap*100:.2f}%")
        print(f"  Dual Bound: {dual_bound_str}")
        print(f"  Primal Bound: {primal_bound_str}")
        print(f"  Result added to summary file")
        
    except Exception as e:
        print(f"  Error solving instance: {str(e)}")
        # Write error result to summary file
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,              # Instance
                n if 'n' in locals() else "Error",  # Vertices 
                m if 'm' in locals() else "Error",  # Edges
                "N/A",               # Objective
                "N/A",                 # DualBound
                "N/A",                 # PrimalBound
                "0.0",                 # Runtime
                "Error",               # Status
            ])
        print(f"  Error result added to summary file")

def main():
    parser = argparse.ArgumentParser(description='Solve Minimum Dominating Set Problem instances using Gurobi')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing problem instances')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='Time limit in seconds (default: 3600)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads to use per instance (default: 0, uses all available)')
    parser.add_argument('--mip_gap', type=float, default=0,
                        help='MIP gap for early termination (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of instances to solve in parallel (default: 1)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.dir + "_sol"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create summary file with header
    summary_file = os.path.join(output_dir, 'summary_results.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Instance", "Vertices", "Edges", "Objective", "DualBound", "PrimalBound", "Runtime", "Status"])
    
    # Get list of problem instances to solve
    instance_files = sorted([f for f in os.listdir(args.dir) if f.endswith('.gr')])
    print(f"Found {len(instance_files)} instances to solve")
    
    # Process instances in batches
    available_cores = multiprocessing.cpu_count()
    effective_batch_size = min(args.batch_size, available_cores, len(instance_files))
    
    # Adjust thread allocation if batch_size > 1
    threads_per_instance = max(1, int(available_cores / effective_batch_size)) if args.threads == 0 else max(1, int(args.threads / effective_batch_size))
    
    if effective_batch_size > 1:
        print(f"Using batch processing with {effective_batch_size} parallel instances")
        print(f"Allocating {threads_per_instance} threads per instance")
        
        # Prepare batch arguments
        batch_args = []
        for filename in instance_files:
            instance_path = os.path.join(args.dir, filename)
            output_path = os.path.join(output_dir, filename)
            batch_args.append((filename, instance_path, output_path, args.time_limit, threads_per_instance, args.mip_gap, summary_file))
        
        # Process batches
        with ProcessPoolExecutor(max_workers=effective_batch_size) as executor:
            executor.map(process_instance, batch_args)
    else:
        # Sequential processing
        for filename in instance_files:
            instance_path = os.path.join(args.dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_instance((filename, instance_path, output_path, args.time_limit, args.threads, args.mip_gap, summary_file))
    
    print(f"Summary of results saved to {summary_file}")

if __name__ == "__main__":
    main()