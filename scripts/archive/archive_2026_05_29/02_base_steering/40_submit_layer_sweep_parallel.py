import os
import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=base_steer_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/base_steer_{trait}.out
#SBATCH --error=log/base_steer_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

# Setup directories
OUT_DIR="exp_steering_layer_analysis/results/{trait}"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_analysis/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

LAYERS=({layers_str})
VALS=({vals_str})

for L in "${{LAYERS[@]}}"; do
    for V in "${{VALS[@]}}"; do
        echo "Running Generation: Trait={trait}, Layer=$L, Val=$V"
        JSONL_OUT="${{OUT_DIR}}/layer_${{L}}_Val${{V}}.jsonl"
        CSV_OUT="${{OUT_DIR}}/scores_layer_${{L}}_Val${{V}}.csv"
        
        if [ ! -f "$JSONL_OUT" ]; then
            "$PYTHON_BIN" scripts/02_base_steering/40_run_layer_sweep.py \\
                --config "$CONFIG" \\
                --vector_bank "$VECTOR_BANK" \\
                --prompts "$PROMPT_IN" \\
                --out_dir "$OUT_DIR" \\
                --axis "{trait}" \\
                --target_layer "$L" \\
                --tau "$V" \\
                --alpha "$V" \\
                --mode both
        else
            echo "  [SKIP] Generation already done: $JSONL_OUT"
        fi
        
        if [ ! -f "$CSV_OUT" ]; then
            "$PYTHON_BIN" scripts/02_base_steering/33_eval_adaptive_steering.py \\
                --input "$JSONL_OUT" \\
                --output "$CSV_OUT" \\
                --axis "{trait}" \\
                --model "$JUDGE_MODEL"
        else
            echo "  [SKIP] Evaluation already done: $CSV_OUT"
        fi
    done
done
"""

def main():
    job_dir = Path("jobs/base_adaptive_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    layers_str = " ".join(map(str, LAYERS))
    vals_str = " ".join(map(str, VALS))
    
    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(
            trait=trait,
            layers_str=layers_str,
            vals_str=vals_str
        )
        
        pbs_file = job_dir / f"run_base_{trait}.pbs"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        
        print(f"Submitting baseline sweep job for {trait}...")
        subprocess.run(["sbatch", str(pbs_file)])

if __name__ == "__main__":
    main()
