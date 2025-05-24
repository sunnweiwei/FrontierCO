#!/bin/bash

# --- Usage Information ---
function show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --instance-dir DIR   Directory containing graph files (default: easy_test_instances)"
    echo "  -p, --parallel NUM       Maximum number of parallel instances (default: 10)"
    echo "  -e, --executable PATH    Path to redumis executable (default: ./redumis)"
    echo "  -t, --time-limit SEC     Time limit in seconds (default: 3600)"
    echo "  -h, --help               Show this help message"
    exit 1
}

# --- Default Configuration ---
REDUMIS_PATH="./redumis"      # Path to redumis executable
MAX_PARALLEL=10               # Maximum number of parallel instances
INSTANCE_DIR="easy_test_instances"
TIME_LIMIT=3600               # Default time limit in seconds

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
        -e|--executable)
            REDUMIS_PATH="$2"
            shift 2
            ;;
        -t|--time-limit)
            TIME_LIMIT="$2"
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

# Create necessary directory
KAMIS_INSTANCE_DIR="${INSTANCE_DIR}_kamis"
RESULTS_DIR="${INSTANCE_DIR}_sol"
mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo " - Redumis Path: $REDUMIS_PATH"
echo " - Instance Directory: $INSTANCE_DIR" 
echo " - Maximum Parallel Processes: $MAX_PARALLEL"
echo " - Results Directory: $RESULTS_DIR"
echo " - Time Limit: $TIME_LIMIT seconds"

# --- Validate settings ---
if [ ! -d "$INSTANCE_DIR" ]; then
    echo "Error: Instance directory '$INSTANCE_DIR' does not exist."
    exit 1
fi

if [ ! -d "$KAMIS_INSTANCE_DIR" ]; then
    echo "Error: KAMIS transformation '$KAMIS_INSTANCE_DIR' does not exist."
    exit 1
fi

if [ ! -x "$REDUMIS_PATH" ]; then
    echo "Warning: Redumis executable '$REDUMIS_PATH' not found or not executable."
    read -p "Continue anyway? (y/n): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# --- Function to solve a single instance ---
solve_instance() {
    local graph_file=$1
    local filename=$(basename "$graph_file")
    local base_filename="${filename%.*}"  # Remove extension
    local base_filename="${base_filename%-sorted}"  # Remove -sorted suffix if present
    local output_file="$RESULTS_DIR/${base_filename}.txt"
    local log_file="$RESULTS_DIR/${base_filename}.log"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting $base_filename" | tee -a "$log_file"
    
    # Run redumis with the graph file
    $REDUMIS_PATH "$graph_file" --time_limit $TIME_LIMIT --output "$output_file" >> "$log_file" 2>&1
    local exit_code=$?
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished $base_filename (Exit code: $exit_code)" | tee -a "$log_file"
    
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
echo "Starting KaMIS redumis solving process with up to $MAX_PARALLEL parallel instances..."

# Get total number of graph files for progress tracking
total_files=$(find "$KAMIS_INSTANCE_DIR" -name "*.graph" | wc -l)
if [ "$total_files" -eq 0 ]; then
    echo "Error: No .graph files found in $KAMIS_INSTANCE_DIR"
    exit 1
fi

echo "Total instances to solve: $total_files"

# Array to keep track of background processes and their associated filenames
declare -a pids
declare -A pid_to_file

# Loop through graph files
for graph_file in "$KAMIS_INSTANCE_DIR"/*.graph; do
    if [ -f "$graph_file" ]; then
        filename=$(basename "$graph_file")
        base_filename="${filename%.*}"  # Remove extension
        base_filename="${base_filename%-sorted}"  # Remove -sorted suffix if present
        output_file="$RESULTS_DIR/${base_filename}.txt"
        log_file="$RESULTS_DIR/${base_filename}.log"
        
        # Skip if already solved
        if [ -f "$output_file" ]; then
            echo "Skipping already solved instance: $base_filename"
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
        echo "Launching solver for $base_filename..."
        solve_instance "$graph_file" &
        
        # Store the PID and filename
        pid=$!
        pids+=($pid)
        pid_to_file[$pid]=$base_filename
        
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

echo "All KaMIS redumis solving processes completed!"
echo "Solutions and logs are available in the $RESULTS_DIR directory"