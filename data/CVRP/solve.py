#!/usr/bin/env python3
"""
CVRP Problem Solver using PyHgese

This script solves Capacitated Vehicle Routing Problem (CVRP) instances using the HGS algorithm.
It processes all input files in a specified directory and saves solutions
to an output directory.

Usage:
    python cvrp_solver_hgs.py --dir input_directory [options]
"""

import os
import sys
import numpy as np
import hygese as hgs
import argparse
import time
import csv
from pathlib import Path
import re

def read_vrp_file(filename):
    """Read CVRP Problem instance from a .vrp file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Parse metadata from header
        n = None  # Number of nodes (including depot)
        capacity = None
        coordinates = []
        demands = []
        depot_idx = 0  # Default depot index
        
        # Read through file sections
        section = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for section headers
            if line.startswith("DIMENSION"):
                n = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line == "NODE_COORD_SECTION":
                section = "coords"
                continue
            elif line == "DEMAND_SECTION":
                section = "demand"
                continue
            elif line == "DEPOT_SECTION":
                section = "depot"
                continue
            elif line == "EOF":
                break
                
            # Parse data based on current section
            if section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0]) - 1  # Convert to 0-indexed
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    # Ensure we have a spot in the array for this node
                    while len(coordinates) <= node_id:
                        coordinates.append(None)
                    
                    coordinates[node_id] = (x, y)
            
            elif section == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0]) - 1  # Convert to 0-indexed
                    demand = int(parts[1])
                    
                    # Ensure we have a spot in the array for this demand
                    while len(demands) <= node_id:
                        demands.append(None)
                    
                    demands[node_id] = demand
            
            elif section == "depot":
                try:
                    depot = int(line)
                    if depot > 0:  # Valid depot ID
                        depot_idx = depot - 1  # Convert to 0-indexed
                except ValueError:
                    pass  # Skip if not a valid depot ID
        
        # Ensure all node coordinates and demands are loaded
        if n is None:
            n = len(coordinates)
        
        # Calculate distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    # Euclidean distance
                    dist_matrix[i, j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    return n, capacity, coordinates, demands, dist_matrix, depot_idx

def solve_cvrp_hgs(n, capacity, coordinates, demands, dist_matrix, depot_idx, time_limit, seed=1, nb_threads=0):
    """Solve the CVRP Problem using PyHgese."""
    # Start timing
    start_time = time.time()
    
    # Prepare data structure for PyHgese
    data = dict()
    data['distance_matrix'] = dist_matrix.tolist()
    data['demands'] = demands
    data['depot'] = depot_idx
    data['vehicle_capacity'] = capacity
    data['service_times'] = np.zeros(len(demands))  # Default service times of 0
    
    # Estimate initial number of vehicles (PyHgese will optimize this)
    total_demand = sum(demands)
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    data['num_vehicles'] = min_vehicles * 2  # Give some buffer
    
    print(f"Problem size: {n} nodes, capacity: {capacity}")
    print(f"Total demand: {total_demand}, initial vehicles: {data['num_vehicles']}")
    
    # Configure HGS algorithm parameters
    algorithm_params = hgs.AlgorithmParameters(
        timeLimit=time_limit,  # Time limit in seconds
        seed=seed,             # Random seed for reproducibility
    )
    
    # Create and configure solver
    solver = hgs.Solver(parameters=algorithm_params, verbose=True)
    
    # Solve the problem
    try:
        result = solver.solve_cvrp(data)
        
        # Extract solution
        obj_val = result.cost
        routes = []
        
        # Process routes
        for i, route in enumerate(result.routes):
            # Insert depot at beginning and end of route
            complete_route = [depot_idx] + route + [depot_idx]
            
            # Calculate route length
            route_length = sum(dist_matrix[complete_route[j]][complete_route[j+1]] for j in range(len(complete_route)-1))
            
            # Calculate demand for this route
            route_demand = sum(demands[j] for j in route)
            
            routes.append({
                'vehicle': i,
                'path': complete_route,
                'length': route_length,
                'demand': route_demand
            })
        
        # Calculate total distance and vehicles used
        total_dist = obj_val
        vehicles_used = len(routes)
        
        runtime = time.time() - start_time
        status = "Optimal" if (runtime < time_limit * 0.99) else "TimeLimit"
        
        return (obj_val, total_dist, routes, vehicles_used, runtime, status, None, obj_val, None)
    
    except Exception as e:
        runtime = time.time() - start_time
        print(f"Error during solving: {str(e)}")
        return None, None, None, None, runtime, "Error", None, None, None

def save_solution(filename, obj_val, total_dist, routes, vehicles_used, n, capacity, runtime, status, 
                 dual_bound, primal_bound, gap):
    """Save solution to a file."""
    sol_filename = os.path.splitext(filename)[0] + "_sol.txt"
    
    with open(sol_filename, 'w') as f:
        # Write problem size, capacity, and optimal objective value
        f.write(f"NAME : {os.path.basename(os.path.splitext(filename)[0])}_solution\n")
        f.write(f"COMMENT : CVRP solution using HGS - obj: {obj_val if obj_val is not None else 'N/A'}\n")
        f.write(f"DIMENSION : {n}\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write(f"DISTANCE : {total_dist if total_dist is not None else 'N/A'}\n")
        f.write(f"VEHICLES : {vehicles_used if vehicles_used is not None else 'N/A'}\n")
        
        # Write routes
        f.write("ROUTES_SECTION\n")
        
        if routes is not None:
            for i, route in enumerate(routes):
                # Convert 0-indexed to 1-indexed for output
                route_str = " ".join(str(node + 1) for node in route['path'])
                f.write(f"{i+1}: {route_str}\n")
                
                # Write route details as comments
                f.write(f"# Route {i+1} - Length: {route['length']:.5f}, Demand: {route['demand']}\n")
        else:
            f.write("# No routes available\n")
        
        # Write runtime and solution status
        f.write("\nSTATISTICS_SECTION\n")
        f.write(f"Runtime : {runtime:.4f}\n")
        f.write(f"Status : {status}\n")
        f.write(f"Objective : {obj_val if obj_val is not None else 'N/A'}\n")
        
        # HGS is a heuristic method and doesn't provide dual bounds or gaps like Gurobi
        f.write(f"DualBound : {dual_bound if dual_bound is not None else 'N/A'}\n")
        f.write(f"PrimalBound : {primal_bound if primal_bound is not None else 'N/A'}\n")
        f.write(f"Gap : {gap*100 if gap is not None else 'N/A'}%\n")
        
        f.write("EOF\n")

def main():
    parser = argparse.ArgumentParser(description='Solve CVRP Problem instances using HGS')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing problem instances')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for solutions (default: input_dir + "_hgs_sol")')
    parser.add_argument('--time_limit', type=float, default=3600.0,
                        help='Time limit in seconds (default: 3600.0)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads to use (default: 0, uses all available)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of instances to solve in parallel (default: 1)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output is None:
        output_dir = args.dir + "_sol"
    else:
        output_dir = args.output
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create summary file with header
    summary_file = os.path.join(output_dir, 'summary_results.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance", "Nodes", "Capacity", "Vehicles", "Objective", "DualBound", "PrimalBound",
            "Runtime", "Status"
        ])
    
    # Get list of instance files to solve
    all_files = sorted([f for f in os.listdir(args.dir) 
                       if f.endswith('.vrp')])
    
    print(f"Found {len(all_files)} instances to solve")
    
    # Process instances in batches
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    # Adjust batch size based on available cores if needed
    available_cores = multiprocessing.cpu_count()
    effective_batch_size = min(args.batch_size, available_cores, len(all_files))
    
    if effective_batch_size > 1:
        print(f"Using batch processing with {effective_batch_size} parallel instances")
        
        # Prepare batch arguments
        batch_args = []
        for filename in all_files:
            instance_path = os.path.join(args.dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Each process gets its own set of threads
            threads_per_instance = max(1, int(available_cores / effective_batch_size)) if args.threads == 0 else max(1, int(args.threads / effective_batch_size))
            
            # Vary seed to get different solutions
            seed = args.seed + all_files.index(filename)
            
            batch_args.append((
                filename, instance_path, output_path, output_dir, args.time_limit, 
                threads_per_instance, seed, summary_file
            ))
        
        # Process batches
        with ProcessPoolExecutor(max_workers=effective_batch_size) as executor:
            executor.map(process_instance, batch_args)
    else:
        # Sequential processing
        for filename in all_files:
            instance_path = os.path.join(args.dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            process_instance((
                filename, instance_path, output_path, output_dir, args.time_limit, 
                args.threads, args.seed, summary_file
            ))
    
    print(f"Summary of results saved to {summary_file}")

def process_instance(args):
    """Process a single instance (for parallel execution)"""
    filename, instance_path, output_path, output_dir, time_limit, threads, seed, summary_file = args
    
    print(f"Solving instance: {filename}")
    
    try:
        # Read problem
        n, capacity, coordinates, demands, dist_matrix, depot_idx = read_vrp_file(instance_path)
        print(f"  Nodes: {n}, Capacity: {capacity}, Depot: {depot_idx+1}")
        
        # Solve problem using HGS
        (obj_val, total_dist, routes, vehicles_used, runtime, status, 
         dual_bound, primal_bound, gap) = solve_cvrp_hgs(
            n, capacity, coordinates, demands, dist_matrix, depot_idx,
            time_limit, seed, threads
        )
        
        # Save solution
        save_solution(
            output_path, obj_val, total_dist, routes, vehicles_used,
            n, capacity, runtime, status, dual_bound, primal_bound, gap
        )
        
        # Format values for summary
        if obj_val is not None:
            obj_val_str = f"{obj_val:.6f}"
        else:
            obj_val_str = "N/A"
        
        # Write result to summary file (with lock to avoid concurrent writes)
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,                           # Instance
                n,                                  # Nodes
                capacity,                           # Capacity
                vehicles_used if vehicles_used is not None else "N/A",  # Vehicles
                obj_val_str,                        # Objective
                '',                                 # DualBound
                obj_val_str,                        # PrimalBound
                f"{runtime:.2f}",                   # Runtime
                status                              # Status
            ])
        
        print(f"  Status: {status}")
        print(f"  Objective: {obj_val_str}")
        if obj_val is not None:
            print(f"  Total Distance: {total_dist:.6f}")
            print(f"  Vehicles Used: {vehicles_used}")
        print(f"  Runtime: {runtime:.2f} seconds")
        print(f"  Solution saved to {os.path.join(output_dir, os.path.splitext(filename)[0] + '_sol.txt')}")
    
    except Exception as e:
        print(f"  Error solving instance: {str(e)}")
        # Write error result to summary file
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,  # Instance
                "Error",   # Nodes
                "Error",   # Capacity
                "Error",   # Vehicles
                "Error",   # Objective
                "Error",   # DualBound
                "Error",   # PrimalBounud
                "0.0",     # Runtime
                "Error"    # Status
            ])

if __name__ == "__main__":
    main()