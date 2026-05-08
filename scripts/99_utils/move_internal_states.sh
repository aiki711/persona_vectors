#!/bin/bash
# Move existing internal state files to new directory structure

BASE_DIR="exp_L10-30"
DEST_ROOT="analysis_results/internal_states"

echo "Moving internal state files from $BASE_DIR to $DEST_ROOT..."

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Base directory $BASE_DIR not found."
    exit 1
fi

# Iterate over model directories
for model_dir in "$BASE_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        # specific check to ignore non-model directories if any
        if [[ "$model_name" == "plots" || "$model_name" == "slopes" ]]; then
            continue
        fi

        # Iterate over result directories (e.g., results_writing, results_advice)
        for result_dir in "$model_dir"/results_*; do
             if [ -d "$result_dir" ]; then
                # Extract dataset name (e.g., writing from results_writing)
                dataset_name=$(basename "$result_dir" | sed 's/results_//')
                
                # Construct destination path
                dest_dir="$DEST_ROOT/$model_name/$dataset_name"
                mkdir -p "$dest_dir"
                
                # Move internal state files
                count=0
                find "$result_dir" -maxdepth 1 -name "*_internal_states.csv" -print0 | while IFS= read -r -d '' file; do
                    mv -v "$file" "$dest_dir/"
                    ((count++))
                done
                # echo "Moved files for $model_name / $dataset_name to $dest_dir"
             fi
        done
    fi
done

echo "Move operation complete."
