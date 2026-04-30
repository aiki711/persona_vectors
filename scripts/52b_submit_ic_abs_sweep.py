import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
# Same VALs as baseline sweep for fair comparison
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ic_abs_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/ic_abs_{trait}.out
#SBATCH --error=log/ic_abs_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_ic_adaptive/results/{trait}"
mkdir -p "$OUT_DIR"

# Same vector bank and prompts as baseline and DLS for fair comparison
CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_analysis/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

LAYERS=({layers_str})
VALS=({vals_str})

for L in "${{LAYERS[@]}}"; do
    for V in "${{VALS[@]}}"; do
        echo "=== IC-Adaptive: Trait={trait}, Layer=$L, Tau=$V ==="
        JSONL_OUT="${{OUT_DIR}}/ic_adapt_layer${{L}}_Tau${{V}}_S1.5.jsonl"
        CSV_OUT="${{OUT_DIR}}/scores_ic_adapt_layer${{L}}_Tau${{V}}.csv"

        # Step 1: Generation
        if [ ! -f "$JSONL_OUT" ]; then
            "$PYTHON_BIN" scripts/50_run_ic_adaptive_steering.py \\
                --config "$CONFIG" \\
                --vector_bank "$VECTOR_BANK" \\
                --prompts "$PROMPT_IN" \\
                --out_dir "$OUT_DIR" \\
                --axis "{trait}" \\
                --target_layer "$L" \\
                --tau "$V" \\
                --ic_scale 1.5
        else
            echo "  [SKIP] Generation done: $JSONL_OUT"
        fi

        # Step 2: Absolute score evaluation (Llama-3)
        if [ ! -f "$CSV_OUT" ]; then
            "$PYTHON_BIN" scripts/60_eval_ic_absolute.py \\
                --input "$JSONL_OUT" \\
                --output "$CSV_OUT" \\
                --axis "{trait}" \\
                --model "$JUDGE_MODEL"
        else
            echo "  [SKIP] Evaluation done: $CSV_OUT"
        fi
    done
done
"""

def main():
    job_dir = Path("jobs/ic_abs_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)

    layers_str = " ".join(map(str, LAYERS))
    vals_str = " ".join(map(str, VALS))

    for trait in TRAITS:
        content = PBS_TEMPLATE.format(trait=trait, layers_str=layers_str, vals_str=vals_str)
        pbs_file = job_dir / f"run_ic_abs_{trait}.pbs"
        pbs_file.write_text(content)
        print(f"Submitting IC-Adaptive (absolute eval) job for {trait}...")
        subprocess.run(["sbatch", str(pbs_file)])

if __name__ == "__main__":
    main()
