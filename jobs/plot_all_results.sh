#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=log/plot_results.out
#SBATCH --error=log/plot_results.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "=== Running Plotting Scripts for All Experiments ==="

echo "1. Plotting All-Layer DLS Heatmaps..."
"$PYTHON_BIN" scripts/04_dyn_layer/72_plot_dyn_layer_heatmaps_all.py

echo "2. Plotting DLS vs Fusion Comparison Heatmaps..."
"$PYTHON_BIN" scripts/04_dyn_layer/75_plot_fusion_comparison.py

echo "=== Plotting Complete ==="
