#!/usr/bin/env python3
"""
Capacitated Facility Location Problem Solver with Batch Processing

This script solves CFLP instances using Gurobi optimizer with batch processing capabilities.
It processes all input files in a specified directory and saves solutions
to an output directory with a '_sol' suffix.

Usage:
    python gurobi_solve.py --dir input_directory [options]
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

def read_cflp_problem(filename):
    """Read Capacitated Facility Location Problem instance from a file.
    
    New format:
    n, m (n=facilities, m=customers)
    b1, f1 (capacity and fixed cost of facility 1)
    b2, f2
    ...
    bn, fn
    d1, d2, d3, ..., dm (customer demands)
    c11, c12, c13, ..., c1m (allocation costs for facility 1 to all customers)
    c21, c22, c23, ..., c2m
    ...
    cn1, cn2, cn3, ..., cnm
    """
    # Read all numbers from the file
    with open(filename, 'r') as f:
        content = f.read()
        # Extract all numbers, ignoring whitespace and empty lines
        all_numbers = [num for num in content.split() if num.strip()]
        
    pos = 0  # Position in the numbers list
    
    # Parse dimensions: n (facilities), m (customers)
    n = int(all_numbers[pos])
    pos += 1
    m = int(all_numbers[pos])
    pos += 1
    
    # Parse facility data: capacity, fixed cost
    capacities = []
    fixed_costs = []
    for _ in range(n):
        if pos + 1 < len(all_numbers):
            capacities.append(float(all_numbers[pos]))
            pos += 1
            fixed_costs.append(float(all_numbers[pos]))
            pos += 1
    
    # Parse customer demands
    demands = []
    for _ in range(m):
        if pos < len(all_numbers):
            demands.append(float(all_numbers[pos]))
            pos += 1
    
    # Parse transportation costs
    trans_costs = []
    for _ in range(n):
        facility_costs = []
        for _ in range(m):
            if pos < len(all_numbers):
                facility_costs.append(float(all_numbers[pos]))
                pos += 1
        trans_costs.append(facility_costs)
    
    # Verify that we have the expected amount of data
    expected_numbers = 2 + 2*n + m + n*m
    if len(all_numbers) < expected_numbers:
        print(f"Warning: File might be incomplete. Expected {expected_numbers} numbers, found {len(all_numbers)}.")
    
    return n, m, capacities, fixed_costs, demands, trans_costs


def solve_cflp(n, m, capacities, fixed_costs, demands, trans_costs, time_limit, threads, mip_gap):
    """Solve the Capacitated Facility Location Problem using Gurobi.
    
    Parameters:
    n (int): Number of facilities
    m (int): Number of customers
    capacities (list): Capacity of each facility
    fixed_costs (list): Fixed cost of opening each facility
    demands (list): Demand of each customer
    trans_costs (list): Transportation costs [facility][customer]
    """
    # Create model
    model = gp.Model("CFLP")
    
    # Set parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)
    model.setParam('MIPGap', mip_gap)
    
    # Create variables
    # y[i] = 1 if facility i is open, 0 otherwise
    y = {}
    for i in range(n):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
    
    # x[i,j] = 1 if customer j is served by facility i, 0 otherwise
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}_{j}")
    
    # Set objective function (minimize total cost: fixed + transportation)
    obj = gp.LinExpr()
    
    # Fixed costs part
    for i in range(n):
        obj += fixed_costs[i] * y[i]
    
    # Transportation costs part
    for i in range(n):
        for j in range(m):
            obj += trans_costs[i][j] * x[i, j]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Add constraint: each customer must be assigned to exactly one facility
    for j in range(m):
        expr = gp.LinExpr()
        for i in range(n):
            expr += x[i, j]
        model.addConstr(expr == 1, f"customer_{j}")
    
    # Add constraint: can only assign customers to open facilities
    for i in range(n):
        for j in range(m):
            model.addConstr(x[i, j] <= y[i], f"open_{i}_{j}")
    
    # Add constraint: capacity of each facility must not be exceeded
    for i in range(n):
        expr = gp.LinExpr()
        for j in range(m):
            expr += demands[j] * x[i, j]
        model.addConstr(expr <= capacities[i] * y[i], f"capacity_{i}")
    
    # Optimize model
    model.optimize()
    
    # Extract solution details
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Get objective value if available
        obj_val = model.ObjVal if model.SolCount > 0 else None
        
        # Extract solution if available
        if model.SolCount > 0:
            facility_status = np.zeros(n, dtype=int)
            assignments = np.zeros((n, m), dtype=float)
            
            for i in range(n):
                facility_status[i] = int(y[i].x > 0.5)
                for j in range(m):
                    assignments[i, j] = x[i, j].x
            
            # Calculate total fixed cost and transportation cost
            fixed_cost = sum(fixed_costs[i] * facility_status[i] for i in range(n))
            transport_cost = sum(trans_costs[i][j] * assignments[i, j] for i in range(n) for j in range(m))
            
            # Calculate capacity usage
            capacity_usage = np.zeros(n)
            for i in range(n):
                capacity_usage[i] = sum(demands[j] * assignments[i, j] for j in range(m))
            
            # Round assignments for integer interpretation
            int_assignments = np.zeros((n, m), dtype=int)
            for i in range(n):
                for j in range(m):
                    int_assignments[i, j] = int(assignments[i, j] > 0.5)
        else:
            facility_status = None
            int_assignments = None
            capacity_usage = None
            fixed_cost = None
            transport_cost = None
        
        # Get dual bound (best lower bound)
        dual_bound = model.ObjBound if hasattr(model, 'ObjBound') else None
        
        # Get primal bound (best feasible solution)
        primal_bound = model.ObjVal if model.SolCount > 0 else None
        
        return obj_val, fixed_cost, transport_cost, facility_status, int_assignments, capacity_usage, model.Runtime, model.status, dual_bound, primal_bound, model.MIPGap if hasattr(model, 'MIPGap') else None
    else:
        return None, None, None, None, None, None, model.Runtime, model.status, None, None, None

def save_solution(filename, obj_val, fixed_cost, transport_cost, facility_status, assignments, capacity_usage, capacities, runtime, n, m, status, dual_bound, primal_bound, gap):
    """Save solution to a file."""
    with open(filename, 'w') as f:
        # Write problem size and optimal objective value
        f.write(f"{n} {m} {int(obj_val) if obj_val is not None else 'N/A'}\n")
        
        # Write fixed cost and transportation cost
        f.write(f"{fixed_cost if fixed_cost is not None else 'N/A'} {transport_cost if transport_cost is not None else 'N/A'}\n")
        
        # Write facility status (open/closed)
        if facility_status is not None:
            f.write(' '.join(map(str, facility_status)) + '\n')
        else:
            f.write("N/A\n")
        
        # Write customer assignments
        if assignments is not None:
            for j in range(m):
                # Find which facility serves customer j
                for i in range(n):
                    if assignments[i, j] == 1:
                        f.write(f"{i}\n")
                        break
                else:
                    f.write("N/A\n")  # If no facility is assigned (infeasible)
        else:
            for j in range(m):
                f.write("N/A\n")
        
        # Write capacity usage for each facility
        if capacity_usage is not None and capacities is not None:
            for i in range(n):
                f.write(f"{capacity_usage[i]} {capacities[i]}\n")
        else:
            for i in range(n):
                f.write(f"N/A {capacities[i]}\n")
        
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
        n, m, capacities, fixed_costs, demands, trans_costs = read_cflp_problem(instance_path)
        print(f"  Facilities: {n}, Customers: {m}")
        
        # Solve problem
        obj_val, fixed_cost, transport_cost, facility_status, assignments, capacity_usage, runtime, status, dual_bound, primal_bound, gap = solve_cflp(
            n, m, capacities, fixed_costs, demands, trans_costs, time_limit, threads, mip_gap
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
            output_path, obj_val, fixed_cost, transport_cost, facility_status, 
            assignments, capacity_usage, capacities, runtime, n, m, status_str, dual_bound, primal_bound, gap
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
                n,                     # Facilities
                m,                     # Customers
                obj_val_str,           # Objective
                dual_bound_str,        # DualBound
                primal_bound_str,      # PrimalBound
                f"{runtime:.1f}",      # Runtime
                status_str             # Status
            ])
        
        print(f"  Status: {status_str}")
        print(f"  Objective: {obj_val_str}")
        if obj_val is not None:
            print(f"  Fixed Cost: {fixed_cost:.2f}")
            print(f"  Transport Cost: {transport_cost:.2f}")
            print(f"  Open Facilities: {sum(facility_status) if facility_status is not None else 'N/A'}/{n}")
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
                n if 'n' in locals() else "Error",  # Facilities 
                m if 'm' in locals() else "Error",  # Customers
                "Error",               # Objective
                "N/A",                 # DualBound
                "N/A",                 # PrimalBound
                "0.0",                 # Runtime
                "Error"                # Status
            ])
        print(f"  Error result added to summary file")

def main():
    parser = argparse.ArgumentParser(description='Solve Capacitated Facility Location Problem instances using Gurobi')
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
    output_dir = args.dir + "_opt_sol"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create summary file with header
    summary_file = os.path.join(output_dir, 'summary_results.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Instance", "Facilities", "Customers", "Objective", "DualBound", "PrimalBound", "Runtime", "Status"])
    
    # Get list of problem instances to solve
    instance_files = sorted([f for f in os.listdir(args.dir) if not f.startswith('.')])
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