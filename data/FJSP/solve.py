#!/usr/bin/env python3
"""
Multi-Problem Optimization Solver

This script solves various optimization problems using different models and solvers.
Supports parallel processing of benchmark instances with configurable parameters.

Usage:
    python solve.py --problem [problem_name] --model [model_type] --time [time_limit] --solver-name [solver] --threads [threads] --input [input_dir]
    
Example:
    python solve.py --problem Parallelmachine --model CP --time 3600 --solver-name CPLEX --threads 10 --input ./easy_test_instanes
"""

import os
import sys
import argparse
import concurrent.futures
import datetime as dt
from pathlib import Path

# Import your baseline models
from baselines import models


def validate_solver_compatibility(problem_name, model_type, solver):
    """Validate solver compatibility with problem and model type."""
    # CP model validation
    if model_type == 'CP':
        if solver not in ['CPLEX', 'Google']:
            return False, f"Solver '{solver}' not supported for CP models. Use 'CPLEX' or 'Google'."
        
        # Google CP solver problem compatibility
        supported_problems = [
            'Non-Flowshop', 'Hybridflowshop', 'Nowaitflowshop', 
            'Jobshop', 'Flexiblejobshop', 'Openshop', 'Parallelmachine'
        ]
        
        if solver == 'Google' and problem_name not in supported_problems:
            return False, f"Google CP solver not supported for problem '{problem_name}'. Supported problems: {supported_problems}"
    
    return True, "Valid configuration"


def process_benchmark(benchmark, model_type, solver, problem_name, computational_time, 
                     n_threads, input_dir, output_dir):
    """Process a single benchmark instance."""
    result_file = os.path.join(
        output_dir, 
        f'result_{model_type}_{problem_name}_{computational_time}_{n_threads}_{benchmark}.txt'
    )
    
    try:
        # Solve the benchmark instance
        n, g, time_taken, lb, ub, gap = models.main(
            int(computational_time), benchmark, problem_name, 
            model_type, solver, n_threads, input_dir, output_dir
        )
        
        # Write successful result
        with open(result_file, 'a') as f:
            f.write(f'\n{problem_name}\t{solver}\t{model_type}\t{benchmark}\t'
                   f'{n}\t{g}\t{lb}\t{ub}\t{gap}\t{time_taken}')
        
        print(f"✓ Solved {benchmark}: LB={lb}, UB={ub}, GAP={gap}, Time={time_taken}s")
        
    except Exception as e:
        # Write error result
        with open(result_file, 'a') as f:
            f.write(f'\n{problem_name}\t{solver}\t{model_type}\t{benchmark}\t'
                   f'ERROR\t{type(e).__name__}: {str(e)}')
        
        print(f"✗ Error solving {benchmark}: {type(e).__name__}: {str(e)}")


def get_benchmark_files(input_dir, file_extensions=None):
    """Get list of benchmark files from input directory."""
    if file_extensions is None:
        file_extensions = ['.txt', '.fjs']
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")
    
    benchmark_files = []
    for file in os.listdir(input_dir):
        if any(file.endswith(ext) for ext in file_extensions):
            benchmark_files.append(file)
    
    return sorted(benchmark_files)


def create_result_headers(output_dir, model_type, problem_name, computational_time, n_threads):
    """Create result files with headers."""
    header = "Problem\tSolver\tModel\tInstance\tN\tG\tLowerBound\tUpperBound\tGAP\tTime"
    
    # Find all potential result files
    benchmark_files = get_benchmark_files(
        output_dir.replace('_sol', ''), 
        ['.txt', '.fjs']
    )
    
    for benchmark in benchmark_files:
        result_file = os.path.join(
            output_dir, 
            f'result_{model_type}_{problem_name}_{computational_time}_{n_threads}_{benchmark}.txt'
        )
        
        # Only write header if file doesn't exist
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write(header)


def main():
    parser = argparse.ArgumentParser(
        description='Job Scheduling Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python solve.py --problem Flexiblejobshop --model CP --time 3600 --solver-name CPLEX --threads 4 --input ./Instances/Flexiblejobshop
        """
    )
    
    # Define arguments using the alternative names that were referenced in the original code
    parser.add_argument('--problem', dest='problem_type', required=True,
                        help='Problem type (e.g., Flexiblejobshop, Parallelmachine, Jobshop)')
    parser.add_argument('--model', dest='model_type', required=True,
                        help='Model type (e.g., CP, MIP)')
    parser.add_argument('--time', dest='computational_time', type=int, required=True,
                        help='Time limit in seconds')
    parser.add_argument('--solver-name', dest='solver_type', required=True,
                        help='Solver name (e.g., CPLEX, Google)')
    parser.add_argument('--threads', dest='n_threads', type=int, default=1,
                        help='Number of threads per instance (default: 1)')
    parser.add_argument('--input', dest='input_dir',
                        help='Input directory path (default: ./Instances/{problem_type})')
    
    # Processing options
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Maximum parallel workers (default: 10)')
    parser.add_argument('--create-headers', action='store_true',
                        help='Create result files with headers before processing')
    parser.add_argument('--file-extensions', nargs='+', default=['.txt', '.fjs'],
                        help='File extensions to look for (default: .txt .fjs)')
    
    args = parser.parse_args()
    
    # Use the alternative argument names directly
    problem_name = args.problem_type
    model_type = args.model_type
    computational_time = args.computational_time
    solver = args.solver_type
    n_threads = args.n_threads
    input_dir = args.input_dir + '_cp'
    
    # Set default input directory if not provided
    if input_dir is None:
        input_dir = f'./easy_test_instances'
    
    # Create output directory
    output_dir = args.input_dir + '_sol'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Problem: {problem_name}")
    print(f"  Model: {model_type}")
    print(f"  Time limit: {computational_time}s")
    print(f"  Solver: {solver}")
    print(f"  Threads per instance: {n_threads}")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max parallel workers: {args.max_workers}")
    
    # Validate solver compatibility
    is_valid, message = validate_solver_compatibility(problem_name, model_type, solver)
    if not is_valid:
        print(f"Error: {message}")
        sys.exit(1)
    
    # Get benchmark files
    try:
        benchmark_files = get_benchmark_files(input_dir, args.file_extensions)
        print(f"Found {len(benchmark_files)} benchmark files")
        
        if not benchmark_files:
            print("No benchmark files found. Check input directory and file extensions.")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create result headers if requested
    if args.create_headers:
        create_result_headers(output_dir, model_type, problem_name, computational_time, n_threads)
        print("Created result file headers")
    
    # Process benchmarks in parallel
    print(f"\nStarting parallel processing at {dt.datetime.now()}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all benchmark tasks
        futures = [
            executor.submit(
                process_benchmark,
                benchmark,
                model_type,
                solver,
                problem_name,
                computational_time,
                n_threads,
                input_dir,
                output_dir
            ) for benchmark in benchmark_files
        ]
        
        # Wait for all tasks to complete and track progress
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            print(f"Progress: {completed}/{len(benchmark_files)} completed")
    
    print(f"\nAll benchmarks processed at {dt.datetime.now()}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()