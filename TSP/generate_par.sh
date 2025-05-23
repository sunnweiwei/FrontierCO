#!/bin/bash
# --- Usage Information ---
function show_usage() {
    echo "Usage: $0 <tsp_directory>"
    echo "  <tsp_directory>: Directory containing TSP files (default: easy_test_instances)"
    exit 1
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
fi

# --- Configuration ---
# Use command line argument if provided, otherwise use default
TSP_DIR="${1:-easy_test_instances}"     # Directory containing your .tsp files

# Define derived directories more robustly
PAR_DIR="${TSP_DIR}_par"                # Directory where parameter files will be saved
RESULTS_DIR="${TSP_DIR}_sol"            # Directory where results will be saved

echo "Using TSP directory: $TSP_DIR"
echo "Parameter files will be saved to: $PAR_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# --- User choice for POPMUSIC ---
echo ""
echo "Choose solver configuration:"
echo "1) Standard LKH configuration"
echo "2) POPMUSIC configuration"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        USE_POPMUSIC=false
        echo "Using standard LKH configuration"
        ;;
    2)
        USE_POPMUSIC=true
        echo "Using POPMUSIC configuration"
        ;;
    *)
        echo "Invalid choice. Using standard LKH configuration by default."
        USE_POPMUSIC=false
        ;;
esac

# Create necessary directories
mkdir -p "$PAR_DIR" "$RESULTS_DIR"

# --- Function to create parameter file for a TSP instance ---
create_par_file() {
    local tsp_file=$1
    local filename=$(basename "$tsp_file")
    local par_file="$PAR_DIR/${filename}.par"
    
    # Create a parameter file for LKH
    cat > "$par_file" << EOF
PROBLEM_FILE = $tsp_file
OUTPUT_TOUR_FILE = $RESULTS_DIR/${filename}.sol
RUNS = 1
SEED = 1
TRACE_LEVEL = 1
EOF

    # Add POPMUSIC parameters only if selected
    if [ "$USE_POPMUSIC" = true ]; then
        cat >> "$par_file" << EOF
CANDIDATE_SET_TYPE = POPMUSIC
POPMUSIC_SOLUTIONS = 1
POPMUSIC_SAMPLE_SIZE = 50
POPMUSIC_TRIALS = 1000
POPMUSIC_MAX_NEIGHBORS = 20
INITIAL_PERIOD = 100
EOF
    fi

    # Add common parameters
    cat >> "$par_file" << EOF
MAX_TRIALS = 1000
TIME_LIMIT = 3600
EOF
    
    echo "Created parameter file: $par_file"
}

# --- Main execution ---
echo "Creating parameter files for TSP instances..."

# Check if TSP directory exists
if [ ! -d "$TSP_DIR" ]; then
    echo "Error: Directory '$TSP_DIR' does not exist"
    show_usage
fi

# Find all TSP files and create parameter files
file_count=0
for tsp_file in "$TSP_DIR"/*; do
    # Skip if not a file
    if [ ! -f "$tsp_file" ]; then
        continue
    fi
    
    create_par_file "$tsp_file"
    ((file_count++))
done

if [ $file_count -eq 0 ]; then
    echo "Warning: No files found in $TSP_DIR"
    exit 0
fi

echo "Successfully created $file_count parameter files in the $PAR_DIR directory."
echo "Parameters set:"
echo " - Time limit: 3600 seconds"
echo " - Single CPU thread per instance (LKH default)"
echo " - Run count: 1"
if [ "$USE_POPMUSIC" = true ]; then
    echo " - Using POPMUSIC candidate set type"
fi
echo ""
echo "You can now review and modify these parameter files before running LKH."