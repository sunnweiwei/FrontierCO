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

def read_graph_file(graph_file_path):
    """Read the graph file to extract additional information if available"""
    graph_info = {
        'Nodes': '',
        'Edges': ''
    }
    
    try:
        with open(graph_file_path, 'r') as f:
            content = f.read()
            nodes = extract_value(r'Nodes:\s*(\d+)', content)
            edges = extract_value(r'Edges:\s*(\d+)', content)
            
            if nodes:
                graph_info['Nodes'] = nodes
            if edges:
                graph_info['Edges'] = edges
    except Exception as e:
        print(f"Error reading graph file {graph_file_path}: {e}")
    
    return graph_info

def process_mis_log(log_path, instance_dir):
    """Process a single MIS log file and extract relevant information"""
    # Extract instance name from log filename
    instance = os.path.basename(log_path).replace('.log', '.mis')
    
    # Read the log file
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None
    
    # Initialize data dictionary
    data = {
        'Instance': instance,
        'Nodes': '',       # Number of nodes in the graph
        'Edges': '',       # Number of edges in the graph
        'Objective': '',   # Best solution (independent set size)
        'DualBound': '',   # Leave DualBound blank as requested
        'PrimalBound': '', # Same as objective for MIS
        'Runtime': '',     # Total runtime
        'Status': 'Unknown'
    }
    
    # Extract number of nodes and edges
    nodes = extract_value(r'\|-Nodes:\s*(\d+)', content)
    edges = extract_value(r'\|-Edges:\s*(\d+)', content)
    
    if nodes:
        data['Nodes'] = nodes
    if edges:
        data['Edges'] = edges
    
    # Look for independent set size in the last line as requested
    lines = content.strip().split('\n')
    for line in reversed(lines):
        set_size = extract_value(r'Independent set has size (\d+)', line)
        if set_size:
            data['Objective'] = set_size
            data['PrimalBound'] = set_size
            break
    
    # If we didn't find it in the last lines, try the usual location
    if not data['Objective']:
        size = extract_value(r'Size:\s*(\d+)', content)
        if size:
            data['Objective'] = size
            data['PrimalBound'] = size
    
    # Extract runtime from "Time found:" field as requested
    time_found = extract_value(r'Time found:\s*([0-9.]+)', content)
    if time_found:
        data['Runtime'] = time_found
    else:
        # Fallback to total time if Time found is not present
        total_time = extract_value(r'Total time:\s*([0-9.]+)', content)
        if total_time:
            data['Runtime'] = total_time
    
    # Determine solution status
    time_limit = extract_value(r'Time limit:\s*(\d+)', content)
    if time_limit and data['Runtime']:
        # Check if it reached timeout
        if float(data['Runtime']) >= float(time_limit) * 0.99:
            data['Status'] = 'TIMEOUT'
        else:
            data['Status'] = 'Completed'
    elif 'Finished' in content and 'Exit code: 0' in content:
        data['Status'] = 'Completed'
    
    # If we have a graph file reference, try to read it for more info
    graph_file = extract_value(r'Filename:\s*([^\n]+)', content)
    if graph_file and (not nodes or not edges):
        # Try to find the graph file
        if os.path.exists(graph_file):
            graph_file_path = graph_file
        else:
            # Try in instance directory
            base_name = os.path.basename(graph_file)
            graph_file_path = os.path.join(instance_dir, base_name)
            
        if os.path.exists(graph_file_path):
            graph_info = read_graph_file(graph_file_path)
            if not data['Nodes'] and graph_info['Nodes']:
                data['Nodes'] = graph_info['Nodes']
            if not data['Edges'] and graph_info['Edges']:
                data['Edges'] = graph_info['Edges']

    return data

def main(instance_dir=None, output_csv=None):
    """Main function to process all logs and generate CSV"""
    if instance_dir is None:
        instance_dir = "easy_test_instances"
    
    # Determine logs directory based on instance_dir
    logs_dir = instance_dir + "_sol"
    
    if output_csv is None:
        output_csv = os.path.join(logs_dir, "summary_results.csv")
    
    # Get all log files
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return
    
    # Process each log file
    results = []
    for log_file in log_files:
        print(f"Processing {os.path.basename(log_file)}...")
        data = process_mis_log(log_file, instance_dir)
        if data:
            results.append(data)
    
    # Sort results by instance name
    results.sort(key=lambda x: x['Instance'])
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Instance', 'Nodes', 'Edges', 'Objective', 'DualBound', 'PrimalBound', 'Runtime', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"MIS summary results saved to {output_csv}")
    print(f"Processed {len(results)} instances")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process MIS log files and generate summary CSV.')
    parser.add_argument('instance_dir', type=str, nargs='?', default="easy_test_instances",
                        help='Directory containing instance files (default: "easy_test_instances")')
    parser.add_argument('output_csv', type=str, nargs='?', default=None,
                        help='Path to save the output CSV file (default: "<instance_dir>_sol/summary_results.csv")')
    
    args = parser.parse_args()
    
    main(instance_dir=args.instance_dir, output_csv=args.output_csv)