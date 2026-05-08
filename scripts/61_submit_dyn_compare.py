import os
import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
METHODS = ["logit_diff", "anti_alignment"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dyn_cmp_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_cmp_{trait}.out
#SBATCH --error=log/dyn_cmp_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer_compare/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_analysis/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=({vals_str})
METHODS=({methods_str})

for M in "${{METHODS[@]}}"; do
    for V in "${{VALS[@]}}"; do
        echo "Running DLS Generation: Method=$M, Trait={trait}, Alpha=$V"
        JSONL_OUT="${{OUT_DIR}}/{trait}/${{M}}_Val${{V}}.jsonl"
        CSV_OUT="${{OUT_DIR}}/{trait}/scores_${{M}}_Val${{V}}.csv"
        
        if [ ! -f "$JSONL_OUT" ]; then
            "$PYTHON_BIN" scripts/61_run_dyn_layer_compare.py \\
                --config "$CONFIG" \\
                --vector_bank "$VECTOR_BANK" \\
                --prompts "$PROMPT_IN" \\
                --out_dir "$OUT_DIR" \\
                --axis "{trait}" \\
                --alpha "$V" \\
                --method "$M"
        else
            echo "  [SKIP] Generation already done: $JSONL_OUT"
        fi
        
        if [ ! -f "$CSV_OUT" ]; then
            "$PYTHON_BIN" scripts/62_eval_dyn_compare.py \\
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
    job_dir = Path("jobs/dyn_layer_compare")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    vals_str = " ".join(map(str, VALS))
    methods_str = " ".join(METHODS)
    
    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(
            trait=trait,
            vals_str=vals_str,
            methods_str=methods_str
        )
        
        pbs_file = job_dir / f"run_compare_{trait}.pbs"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        
        print(f"Submitting DLS comparison job for {trait}...")
        subprocess.run(["sbatch", str(pbs_file)])

if __name__ == "__main__":
    main()
