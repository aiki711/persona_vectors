#!/bin/bash
#SBATCH --job-name=test_load_70b
#SBATCH --partition=GPU-1A
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=log/test_load_70b.out
#SBATCH --error=log/test_load_70b.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Attempting to load Llama-3-70B-Instruct in BF16 on 2x A100..."

"$PYTHON_BIN" -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from persona_vectors.live_axes import load_model_and_tokenizer

try:
    print('Starting loading process...')
    model, tokenizer = load_model_and_tokenizer('meta-llama/Meta-Llama-3-70B-Instruct', quant=None)
    print('Successfully loaded the model and tokenizer!')
    print('Memory footprint:', model.get_memory_footprint())
except Exception as e:
    print('Failed to load model:', e)
    raise e
"
