#!/bin/bash

# --- Usage Information ---
function show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --instance-dir DIR   Directory containing .par files (default: easy_test_instances)"
    echo "  -p, --parallel NUM       Maximum number of parallel instances (default: 10)"
    echo "  -l, --lkh-path PATH      Path to LKH executable (default: ./LKH-3.0.13/LKH)"
    echo "  -h, --help               Show this help message"
    exit 1
}

# --- Default Configuration ---
LKH_PATH="./LKH-3.0.13/LKH"  # Update this with your actual path
MAX_PARALLEL=10              # Maximum number of parallel instances
INSTANCE_DIR="easy_test_instances"

# --- Parse command line arguments ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--instance-dir)
            INSTANCE_DIR="$2"
            shift 2
            ;;
        -p|--parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -l|--lkh-path)
            LKH_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# --- Setup directories ---
PAR_DIR="${INSTANCE_DIR}_par"
RESULTS_DIR="${INSTANCE_DIR}_sol"

# Create necessary directory
mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo " - LKH Path: $LKH_PATH"
echo " - Par Directory: $PAR_DIR" 
echo " - Maximum Parallel Processes: $MAX_PARALLEL"
echo " - Results Directory: $RESULTS_DIR" # Both solutions and logs

# --- Validate settings ---
if [ ! -d "$PAR_DIR" ]; then
    echo "Error: Paramerer directory '$PAR_DIR' does not exist."
    exit 1
fi

if [ ! -x "$LKH_PATH" ]; then
    echo "Warning: LKH executable '$LKH_PATH' not found or not executable."
    read -p "Continue anyway? (y/n): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# --- Function to solve a single instance ---
solve_instance() {
    local par_file=$1
    local filename=$(basename "$par_file" .par)
    local log_file="$RESULTS_DIR/${filename}.log"
    
    # Run LKH with the parameter file
    $LKH_PATH "$par_file" >> "$log_file" 2>&1
    local exit_code=$?
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished $filename (Exit code: $exit_code)" | tee -a "$log_file"
    
    # Return exit code
    return $exit_code
}

# --- Function to display progress ---
show_progress() {
    local total=$1
    local completed=0
    local running=${#pids[@]}
    
    # Count completed files
    for result_file in "$RESULTS_DIR"/*.log; do
        if [ -f "$result_file" ]; then
            ((completed++))
        fi
    done
    
    local percent=$((completed * 100 / total))
    echo "Progress: $completed/$total completed ($percent%), $running currently running"
}

# --- Main execution ---
echo "Starting LKH solving process with up to $MAX_PARALLEL parallel instances..."

# Get total number of par files for progress tracking
total_files=$(find "$PAR_DIR" -name "*.par" | wc -l)
if [ "$total_files" -eq 0 ]; then
    echo "Error: No .par files found in $PAR_DIR"
    exit 1
fi

echo "Total instances to solve: $total_files"

# Array to keep track of background processes and their associated filenames
declare -a pids
declare -A pid_to_file

# Loop through parameter files
for par_file in "$PAR_DIR"/*.par; do
    if [ -f "$par_file" ]; then
        filename=$(basename "$par_file" .par)
        output_file="$RESULTS_DIR/${filename}.log"
        
        # Skip if already solved
        if [ -f "$output_file" ]; then
            echo "Skipping already solved instance: $filename"
            continue
        fi
        
        # Wait if we've reached max parallel processes
        while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
            # Display progress
            show_progress $total_files
            
            # Check for completed processes
            for i in "${!pids[@]}"; do
                if ! kill -0 ${pids[$i]} 2>/dev/null; then
                    echo "Process for ${pid_to_file[${pids[$i]}]} completed"
                    unset pid_to_file[${pids[$i]}]
                    unset pids[$i]
                fi
            done
            
            # Re-index the array
            pids=("${pids[@]}")
            
            # If still at max, sleep briefly
            if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
                sleep 2
            fi
        done
        
        # Start a new process
        echo "Launching solver for $filename..."
        solve_instance "$par_file" &
        
        # Store the PID and filename
        pid=$!
        pids+=($pid)
        pid_to_file[$pid]=$filename
        
        # Brief pause to prevent race conditions
        sleep 0.5
    fi
done

# Wait for all remaining processes to complete
echo "Waiting for all processes to complete..."
while [ ${#pids[@]} -gt 0 ]; do
    # Display progress
    show_progress $total_files
    
    # Check for completed processes
    for i in "${!pids[@]}"; do
        if ! kill -0 ${pids[$i]} 2>/dev/null; then
            echo "Process for ${pid_to_file[${pids[$i]}]} completed"
            unset pid_to_file[${pids[$i]}]
            unset pids[$i]
        fi
    done
    
    # Re-index the array
    pids=("${pids[@]}")
    
    # Sleep briefly before checking again
    if [ ${#pids[@]} -gt 0 ]; then
        sleep 5
    fi
done

# Final progress report
show_progress $total_files

echo "All LKH solving processes completed!"
echo "Solutions and logs are available in the $RESULTS_DIR directory"
