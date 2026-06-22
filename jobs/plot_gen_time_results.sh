#!/bin/bash
#SBATCH --job-name=plot_gen_time_results
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --output=log/plot_gen_time_results.out
#SBATCH --error=log/plot_gen_time_results.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# Activate virtual environment
source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "=== Running Generation-Time DLS Plotting Script ==="

"$PYTHON_BIN" scripts/04_dyn_layer/123_plot_gen_time_results.py \
    --results_dir exp_steering_dyn_gen_time_raw/results \
    --out_dir exp_steering_dyn_gen_time_raw/figures \
    --artifact_dir /home/s2550009/.gemini/antigravity-ide/brain/316d92fc-a09f-45ab-a84d-a1a4060ccdb9/images \
    --title_prefix "Raw_GenTime"

echo "=== Plotting Complete ==="
