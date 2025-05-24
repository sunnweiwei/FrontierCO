import os
import csv
import glob
import argparse

def process_fjsp_log(log_path):
    """Process a single FJSP result file and extract relevant information"""
    
    try:
        with open(log_path, 'r') as f:
            content = f.read().strip()
            
        # Split the content by tabs or multiple spaces
        parts = [part.strip() for part in content.split() if part.strip()]
        
        # Check if we have enough parts
        if len(parts) < 10:
            print(f"Error: Not enough data in {log_path}")
            return None
        
        # Extract required data
        # Format: Flexiblejobshop CPLEX CP instance_name jobs machines dual_bound primal_bound gap runtime
        instance = parts[3]  # instance name (e.g., lar01_1.txt)
        jobs = parts[4]      # number of jobs
        machines = parts[5]  # number of machines
        dual_bound = parts[6]   # dual bound
        primal_bound = parts[7] # primal bound
        runtime = parts[9]   # runtime
        
        # Determine status based on gap (parts[8])
        try:
            gap = float(parts[8])
            status = "Optimal" if gap == 0.0 else "TIMEOUT"
        except ValueError:
            status = "Unknown"
        
        # Create data dictionary
        data = {
            'Instance': instance,
            'Jobs': jobs,
            'Machines': machines,
            'Objective': primal_bound,
            'DualBound': dual_bound,
            'PrimalBound': primal_bound,
            'Runtime': runtime,
            'Status': status
        }
        
        return data
        
    except Exception as e:
        print(f"Error processing {log_path}: {e}")
        return None

def main(result_dir=None, output_csv=None):
    """Main function to process all FJSP result files and generate CSV"""
    
    if output_csv is None:
        output_csv = os.path.join(result_dir, "fjsp_results.csv")
    
    # Get all result files that start with 'result'
    result_files = glob.glob(os.path.join(result_dir, "result*"))
    
    if not result_files:
        print(f"No result files found in {result_dir}")
        return
    
    # Process each result file
    results = []
    for result_file in result_files:
        print(f"Processing {os.path.basename(result_file)}...")
        data = process_fjsp_log(result_file)
        if data:
            results.append(data)
    
    # Sort results by instance name
    results.sort(key=lambda x: x['Instance'])
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Jobs', 'Machines', 'Objective' ,'DualBound', 'PrimalBound', 'Runtime', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"FJSP summary results saved to {output_csv}")
    print(f"Processed {len(results)} instances")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process FJSP result files and generate summary CSV.')
    parser.add_argument('result_dir', type=str, default=None,
                        help='Directory containing instance files')
    parser.add_argument('output_csv', type=str, default=None,
                        help='Path to save the output CSV file (default: "<result_dir>/fjsp_results.csv")')
    
    args = parser.parse_args()
    
    main(result_dir=args.result_dir, output_csv=args.output_csv)