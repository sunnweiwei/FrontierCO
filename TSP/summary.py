import os
import re
import csv
import glob
import argparse

def extract_value(pattern, text, default="", group=1):
    """Extract value using regex pattern"""
    match = re.search(pattern, text)
    if match:
        return match.group(group)
    return default

def read_tsp_file(tsp_file_path):
    """Read the TSP file to extract additional information"""
    tsp_info = {
        'NAME': '',
        'TYPE': '',
        'COMMENT': '',
        'DIMENSION': '',
        'EDGE_WEIGHT_TYPE': '',
        'DISPLAY_DATA_TYPE': ''
    }
    
    try:
        with open(tsp_file_path, 'r') as f:
            # Read until NODE_COORD_SECTION or EOF
            for line in f:
                if line.strip() == 'NODE_COORD_SECTION' or line.strip() == 'EDGE_WEIGHT_SECTION':
                    break
                
                # Look for key: value pairs
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key in tsp_info:
                        tsp_info[key] = value
    except Exception as e:
        print(f"Error reading TSP file {tsp_file_path}: {e}")
    
    return tsp_info

def process_tsp_log(log_path, instance_dir):
    """Process a single LKH log file for TSP instance and extract relevant information"""
    instance = os.path.basename(log_path).replace('.log', '')
    
    # Read the log file
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None
    
    # Initialize data dictionary with TSP-specific fields
    data = {
        'Instance': instance,
        'Nodes': '',       # Number of cities/nodes in the TSP
        'Objective': '',   # Best tour length
        'DualBound': '',   # Lower bound if available
        'PrimalBound': '', # Same as objective for TSP
        'Runtime': '3600',     # Total runtime
        'Status': 'Unknown'
    }
    
    # Try to extract the problem file path from the log
    problem_file = extract_value(r'PROBLEM_FILE\s*=\s*"([^"]+)"', content)
    if not problem_file:
        problem_file = extract_value(r'PROBLEM_FILE\s*=\s*([^\s]+)', content)
    
    # If we have a problem file reference, try to read it for more info
    tsp_info = {}
    if problem_file:
        # Check if file exists
        if os.path.exists(problem_file):
            tsp_file_path = problem_file
        else:
            # Try relative to current directory
            tsp_file_path = os.path.join(os.getcwd(), problem_file)
            if not os.path.exists(tsp_file_path):
                # Try in instance directory
                base_name = os.path.basename(problem_file)
                tsp_file_path = os.path.join(instance_dir, base_name)
        
        if os.path.exists(tsp_file_path):
            tsp_info = read_tsp_file(tsp_file_path)
    
    # Extract problem dimension (number of nodes/cities)
    # First check if we got it from the TSP file
    if tsp_info.get('DIMENSION'):
        data['Nodes'] = tsp_info['DIMENSION']
    else:
        # Try to extract from log
        dimension = extract_value(r'DIMENSION\s*[=:]\s*(\d+)', content)
        if not dimension:
            dimension = extract_value(r'dimension\s*:?\s*(\d+)', content)
            if not dimension:
                # Try another pattern typically found in LKH logs
                nodes_match = re.search(r'Reading [^:]+: "[^"]+" \.\.\. (\d+) nodes', content)
                if nodes_match:
                    dimension = nodes_match.group(1)
        
        if dimension:
            data['Nodes'] = dimension
    
    # Calculate edges for a complete graph
    if data['Nodes']:
        try:
            n = int(data['Nodes'])
        except ValueError:
            pass
    
    # Extract best tour length (objective/primal bound)
    best_cost = None
    # Try to get from the final summary
    cost_min = extract_value(r'Cost\.min\s*=\s*([0-9.]+)', content)
    if cost_min:
        best_cost = cost_min
    else:
        # Try to get from last reported tour
        cost_matches = re.findall(r'Cost\s*=\s*([0-9.]+)', content)
        if cost_matches:
            best_cost = cost_matches[-1]  # Get the last reported cost
    
    if best_cost:
        data['Objective'] = best_cost
        data['PrimalBound'] = best_cost
    
    # Extract lower bound (dual bound)
    lower_bound = extract_value(r'Lower bound\s*=\s*([0-9.]+)', content)
    if lower_bound:
        data['DualBound'] = lower_bound
    
    # Extract runtime
    runtime = extract_value(r'Time\.total\s*=\s*([0-9.]+)', content)
    if not runtime:
        runtime = extract_value(r'Time\.max\s*=\s*([0-9.]+)', content)
    if not runtime:
        # Try to extract from the final line of a run
        runtime = extract_value(r'Time\s*=\s*([0-9.]+)\s*sec', content)
    
    if runtime:
        data['Runtime'] = runtime
    
    # Determine solution status - check for explicit time limit message first
    if '*** Time limit exceeded ***' in content:
        data['Status'] = 'TIMEOUT'
    else:
        time_limit = extract_value(r'TIME_LIMIT\s*=\s*([0-9.]+)', content)
        optimum_reached = 'Optimum reached' in content or 'OPTIMUM REACHED' in content
        gap_zero = re.search(r'Gap\.avg\s*=\s*0\.0000%', content) is not None
        
        if time_limit and runtime and float(runtime) >= float(time_limit) * 0.99:
            data['Status'] = 'TIMEOUT'
        else:
            data['Status'] = 'Completed'

    return data

def main(instance_dir=None, output_csv=None):
    """Main function to process all logs and generate CSV"""
    if instance_dir is None:
        instance_dir = "hard_test_instances"
    
    # Determine logs directory based on instance_dir
    logs_dir = instance_dir + "_sol"
    
    if output_csv is None:
        output_csv = os.path.join(logs_dir, "raw_results.csv")
    
    # Get all log files
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return
    
    # Process each log file
    results = []
    for log_file in log_files:
        print(f"Processing {os.path.basename(log_file)}...")
        data = process_tsp_log(log_file, instance_dir)
        if data:
            results.append(data)
    
    # Sort results by instance name
    results.sort(key=lambda x: x['Instance'])
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Nodes', 'Objective', 'DualBound', 'PrimalBound', 'Runtime', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"TSP summary results saved to {output_csv}")
    print(f"Processed {len(results)} instances")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process TSP log files and generate summary CSV.')
    parser.add_argument('instance_dir', type=str, default="hard_test_instances",
                        help='Directory containing instance files (default: "hard_test_instances")')
    parser.add_argument('output_csv', type=str, default=None,
                        help='Path to save the output CSV file (default: "<instance_dir>_sol/summary_results.csv")')
    
    args = parser.parse_args()
    
    main(instance_dir=args.instance_dir, output_csv=args.output_csv)
