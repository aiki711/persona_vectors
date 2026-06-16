#!/bin/bash
#SBATCH --job-name=dls_diagnose
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:30:00
#SBATCH --output=log/diagnose.out
#SBATCH --error=log/diagnose.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Running DLS diagnosis script..."
"$PYTHON_BIN" scratch/diagnose_dls_alignment.py --axis extraversion --num_prompts 5
echo "Done."
