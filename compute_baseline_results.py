import pandas as pd
import numpy as np
import argparse

def compute_metrics(reference_csv_path, result_csv_path):
    """
    Compute the primal gap and runtime metrics from two CSV files
    
    Args:
        reference_csv_path (str): Path to the reference CSV file with primal bounds
        result_csv_path (str): Path to the result CSV file with objectives and times
        
    Returns:
        tuple: (arithmetic_mean_gap, geometric_mean_runtime)
    """
    # Read the CSV files
    reference_df = pd.read_csv(reference_csv_path)
    result_df = pd.read_csv(result_csv_path)
    
    # Convert empty strings to NaN for easier handling
    if 'Objective' in result_df.columns:
        result_df['Objective'] = pd.to_numeric(result_df['Objective'], errors='coerce')
    if 'Time' in result_df.columns:
        result_df['Time'] = pd.to_numeric(result_df['Time'], errors='coerce')
    
    # Create a dictionary for fast lookup of primal bounds
    primal_bound_dict = dict(zip(reference_df['Instance'], reference_df['PrimalBound']))
    
    # Calculate primal gap for each instance
    gaps = []
    times = []
    
    for _, row in result_df.iterrows():
        instance = row['Instance']
        objective = row.get('Objective', None)
        time = row.get('Time', None)
        
        # Check if instance exists in reference data
        primal_bound = primal_bound_dict.get(instance)
        
        # Handle missing data: if instance not in reference, or objective/time is NaN
        if primal_bound is None or pd.isna(objective) or pd.isna(time):
            gap = 1.0
            time_value = 3600.0
        else:
            # Calculate gap using formula: |obj-pb|/max(|obj|,|pb|)
            numerator = abs(float(objective) - float(primal_bound))
            denominator = max(abs(float(objective)), abs(float(primal_bound)))
            
            if denominator == 0:
                gap = 1.0
            else:
                gap = numerator / denominator
            
            time_value = float(time)
        
        gaps.append(gap)
        times.append(time_value)
    if len(gaps) < len(reference_df):
        gaps.extend([1 for i in range(len(reference_df)-len(gaps))])
        times.extend([3600 for i in range(len(reference_df)-len(gaps))])

    # Calculate arithmetic mean of gaps
    arithmetic_mean_gap = np.mean(gaps)
    
    # Calculate geometric mean of time
    # Using log approach to avoid overflow with large numbers
    geometric_mean_time = np.exp(np.mean(np.log(np.array(times) + 0.1)))
    
    return arithmetic_mean_gap, geometric_mean_time


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Compute primal gap and runtime metrics from two CSV files')
    parser.add_argument('reference_csv_path', type=str, help='Path to the reference CSV file with primal bounds')
    parser.add_argument('result_csv_path', type=str, help='Path to the result CSV file with objectives and times')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Compute metrics
    arith_mean_gap, geo_mean_runtime = compute_metrics(
        args.reference_csv_path, args.result_csv_path)
    
    # Print results
    if args.verbose:
        print("Detailed Results:")
        print(detailed_results)
        print("\n")
    
    print(f"Arithmetic Mean of Primal Gap: {arith_mean_gap:.6f}")
    print(f"Geometric Mean of Runtime: {geo_mean_runtime:.6f}")


if __name__ == "__main__":
    main()