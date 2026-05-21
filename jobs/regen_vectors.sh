#!/bin/bash
#SBATCH --job-name=regen_vectors
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=log/regen_vectors.out
#SBATCH --error=log/regen_vectors.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "=== Regenerating vector bank with midpoint ==="
echo "  Script: scripts/01_vectors/30b_train_mean_diff.py"
echo "  Config: config/mistral_7b.yaml"
echo "  Output: vectors/mean_diff_vectors.npz"
echo ""

"$PYTHON_BIN" scripts/01_vectors/30b_train_mean_diff.py \
    --config config/mistral_7b.yaml \
    --out_dir vectors/

echo "=== Done. Verifying midpoint keys ==="
"$PYTHON_BIN" -c "
import numpy as np
d = np.load('vectors/mean_diff_vectors.npz')
mp_keys = [k for k in d.files if 'midpoint' in k]
print(f'Total keys: {len(d.files)}, midpoint keys: {len(mp_keys)}')
print('Example:', mp_keys[:4])
"
