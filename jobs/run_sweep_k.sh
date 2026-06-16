#!/bin/bash
#SBATCH --job-name=sweep_k
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:30:00
#SBATCH --output=log/sweep_k.out
#SBATCH --error=log/sweep_k.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "=== Sweep K probe filter dimensions ==="

for k in 250 500 1000 2000 3000; do
    echo ""
    echo "========================================"
    echo "Running for K = $k..."
    echo "========================================"
    
    # Train probe filters for this k
    "$PYTHON_BIN" scripts/01_vectors/36_train_probe_filters.py --k $k
    
    # Run diagnosis
    "$PYTHON_BIN" scratch/diagnose_dls_alignment.py --axis extraversion --num_prompts 5
done

echo "Done."
