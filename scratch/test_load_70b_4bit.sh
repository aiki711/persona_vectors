#!/bin/bash
#SBATCH --job-name=test_load_70b_4bit
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=00:30:00
#SBATCH --output=log/test_load_70b_4bit.out
#SBATCH --error=log/test_load_70b_4bit.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${PYTHONPATH:-}"
export PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Attempting to load Llama-3-70B-Instruct in 4bit on 1x A40 (GPU-1)..."

"$PYTHON_BIN" -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from persona_vectors.live_axes import load_model_and_tokenizer

try:
    print('Starting loading process (4bit)...')
    from persona_vectors.live_axes import _resolve_hf_token
    import persona_vectors.live_axes as la
    import pathlib
    p_file = la.__file__
    proj_root = pathlib.Path(p_file).resolve().parent.parent.parent
    tokfile = proj_root / \".hf_token\"
    print('DEBUG: la.__file__ =', p_file)
    print('DEBUG: proj_root =', proj_root)
    print('DEBUG: tokfile =', tokfile)
    print('DEBUG: tokfile exists? =', tokfile.exists())
    if tokfile.exists():
        print('DEBUG: tokfile content prefix =', tokfile.read_text().strip()[:10])
    tok = _resolve_hf_token()
    print('Resolved token prefix:', tok[:10] if tok else None)
    model, tokenizer = load_model_and_tokenizer('meta-llama/Meta-Llama-3-70B-Instruct', quant='4bit')
    print('Successfully loaded the model and tokenizer in 4bit!')
    print('Memory footprint:', model.get_memory_footprint())
except Exception as e:
    print('Failed to load model:', e)
    raise e
"
