import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dls_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dls_{trait}.out
#SBATCH --error=log/dls_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer/results/{trait}"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_analysis/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=({vals_str})

for V in "${{VALS[@]}}"; do
    echo "=== DLS: Trait={trait}, Alpha=$V ==="
    JSONL_OUT="${{OUT_DIR}}/dyn_Val${{V}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/scores_dyn_Val${{V}}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/59_run_dynamic_layer_steering.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$V"
    else
        echo "  [SKIP] Generation done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/59_eval_dynamic_layer.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation done: $CSV_OUT"
    fi
done
"""


def main():
    job_dir = Path("jobs/dyn_layer_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(map(str, VALS))

    for trait in TRAITS:
        content = PBS_TEMPLATE.format(trait=trait, vals_str=vals_str)
        pbs_file = job_dir / f"run_dls_{trait}.pbs"
        pbs_file.write_text(content)

        print(f"Submitting DLS job for {trait}...")
        subprocess.run(["sbatch", str(pbs_file)])


if __name__ == "__main__":
    main()
