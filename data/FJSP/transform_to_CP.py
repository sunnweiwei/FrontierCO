import os
import glob
import argparse

def convert_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
    # Parse first line to get number of jobs and machines
    parts = lines[0].strip().split()
    n_jobs = int(parts[0])
    n_machines = int(parts[1])
    
    # Initialize the output data
    operations_per_job = []
    processing_times = []
    
    line_index = 1
    
    # Process each job
    for j in range(n_jobs):
        parts = lines[line_index].strip().split()
        
        n_operations = int(parts[0])
        operations_per_job.append(n_operations)
        
        job_operations = []
        
        part_idx = 1
        # Process each operation for this job
        for k in range(n_operations):
            # Initialize processing times for this operation (0 for all machines)
            operation_times = [0] * n_machines
                        
            n_eligible_machines = int(parts[part_idx])
            part_idx += 1
            # Process each eligible machine for this operation
            for m in range(n_eligible_machines):
                machine_idx = int(parts[part_idx])-1       # Machine index
                proc_time = int(parts[part_idx+1])      # Processing time
                part_idx += 2
                operation_times[machine_idx] = proc_time
            
            job_operations.append(operation_times)
        
        processing_times.append(job_operations)
        line_index += 1
    # Write the converted data to the output file
    with open(output_file_path, 'w') as f:
        # First line: number of jobs
        f.write(f"{n_jobs}\n")
        
        # Second line: number of machines
        f.write(f"{n_machines}\n")
        
        # Third line: number of operations for each job
        f.write(" ".join(map(str, operations_per_job)) + "\n")
        
        # For each job and operation: processing times
        for j in range(n_jobs):
            for k in range(operations_per_job[j]):
                f.write(" ".join(map(str, processing_times[j][k])) + "\n")
    
    print(f"Converted {input_file_path} to {output_file_path}")

def batch_convert(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all .txt files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.txt")) + glob.glob(os.path.join(input_dir, "*.fjs"))
    
    for input_file in input_files:
        # Create output file path with the same basename
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, basename)
        
        convert_file(input_file, output_file)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert scheduling problem files to a different format')
    parser.add_argument('input_dir', help='Input directory containing files to convert')
    parser.add_argument('output_dir', help='Output directory for converted files')
    
    args = parser.parse_args()
    
    batch_convert(args.input_dir, args.output_dir)