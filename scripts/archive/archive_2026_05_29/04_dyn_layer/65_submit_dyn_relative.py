"""
65_submit_dyn_relative.py

改良版 anti_alignment（relative_anti_alignment）実験のジョブ投入スクリプト。
性格特性ごとに SLURM ジョブを生成・投入する。

実験フォルダ: exp_steering_dyn_layer_relative/
"""

import os
import subprocess
from pathlib import Path

TRAITS  = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS    = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dyn_rel_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_rel_{trait}.out
#SBATCH --error=log/dyn_rel_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer_relative/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
# midpoint が含まれる再生成済みのベクトルバンクを使用
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=({vals_str})

for V in "${{VALS[@]}}"; do
    echo "Running Relative Anti-Alignment DLS: Trait={trait}, Alpha=$V"
    JSONL_OUT="${{OUT_DIR}}/{trait}/relative_anti_alignment_Val${{V}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_relative_anti_alignment_Val${{V}}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/65_run_dyn_layer_relative.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$V" \\
            --direction "high"
    else
        echo "  [SKIP] Generation already done: $JSONL_OUT"
    fi

    if [ ! -f "$CSV_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/62_eval_dyn_compare.py \\
            --input "$JSONL_OUT" \\
            --output "$CSV_OUT" \\
            --axis "{trait}" \\
            --model "$JUDGE_MODEL"
    else
        echo "  [SKIP] Evaluation already done: $CSV_OUT"
    fi
done
"""


def main():
    job_dir = Path("jobs/dyn_layer_relative")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    vals_str = " ".join(map(str, VALS))

    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(
            trait=trait,
            vals_str=vals_str,
        )

        pbs_file = job_dir / f"run_relative_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)

        print(f"Submitting relative DLS job for {trait}...")
        result = subprocess.run(["sbatch", str(pbs_file)], capture_output=True, text=True)
        print(f"  {result.stdout.strip()} {result.stderr.strip()}")


if __name__ == "__main__":
    main()
