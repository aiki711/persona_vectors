#!/bin/bash
#SBATCH --job-name=test_load_70b_8bit
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=log/test_load_70b_8bit.out
#SBATCH --error=log/test_load_70b_8bit.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
export PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Attempting to load Llama-3-70B-Instruct in 8bit on 2x A40 (GPU-1)..."

"$PYTHON_BIN" -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from persona_vectors.live_axes import load_model_and_tokenizer

try:
    print('Starting loading process (8bit)...')
    model, tokenizer = load_model_and_tokenizer('meta-llama/Meta-Llama-3-70B-Instruct', quant='8bit')
    print('Successfully loaded the model and tokenizer in 8bit!')
    print('Memory footprint:', model.get_memory_footprint())
except Exception as e:
    print('Failed to load model:', e)
    raise e
"
