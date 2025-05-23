# Updated solve.sh

#!/bin/bash

# Usage: ./solve.sh <input_folder> <solver_binary> <param_file> [max_parallel_jobs]
# Example: ./solve.sh valid_instances /path/to/stp.linux.x86_64.gnu.opt.cpx write.set 8

INPUT_FOLDER=$1
SOLVER_BIN=$2
PARAM_FILE=$3
MAX_JOBS=${4:-$(nproc)}
SOLUTION_FOLDER="${INPUT_FOLDER}_sol"

if [ -z "$INPUT_FOLDER" ] || [ -z "$SOLVER_BIN" ] || [ -z "$PARAM_FILE" ]; then
  echo "Usage: $0 <input_folder> <solver_binary> <param_file> [max_parallel_jobs]"
  exit 1
fi

if ! command -v parallel &> /dev/null; then
  echo "GNU Parallel is required but not installed."
  exit 1
fi

if [ ! -x "$SOLVER_BIN" ]; then
  echo "Error: solver binary '$SOLVER_BIN' not found or not executable."
  exit 1
fi

if [ ! -f "$PARAM_FILE" ]; then
  echo "Error: parameter file '$PARAM_FILE' not found."
  exit 1
fi

mkdir -p "$SOLUTION_FOLDER"
export SOLUTION_FOLDER
export PARAM_FILE
export SOLVER_BIN

process_file() {
  local file=$1
  local filename=$(basename "$file" .stp)
  echo "Processing $file..."
  "$SOLVER_BIN" -f "$file" -s "$PARAM_FILE" -l "${filename}_output.log"
  if [ $? -eq 0 ]; then
    find . -maxdepth 1 -name "${filename}.stplog" -exec mv {} "$SOLUTION_FOLDER/" \;
    mv "${filename}_output.log" "$SOLUTION_FOLDER/" 2>/dev/null
    echo "✓ $filename"
  else
    echo "✗ $filename"
  fi
}

export -f process_file

STP_FILES=$(find "$INPUT_FOLDER" -type f -name "*.stp")
TOTAL=$(echo "$STP_FILES" | wc -l)

echo "Found $TOTAL .stp files in $INPUT_FOLDER"
echo "Using solver: $SOLVER_BIN"
echo "Using parameter file: $PARAM_FILE"
echo "Launching up to $MAX_JOBS parallel jobs..."

echo "$STP_FILES" | parallel -j "$MAX_JOBS" --line-buffer process_file {}

echo "All done. Logs and .stplog files are in $SOLUTION_FOLDER"
