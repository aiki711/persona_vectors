#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 71_submit_dyn_all.py
#
# 全32層（0〜31）を対象にした動的レイヤー選択（DLS）実験のジョブ投入スクリプト。
# キャリブレーションジョブを投入し、その完了後に5つの特性の評価ジョブを依存関係付きで投入する。
#
# 実験フォルダ: exp_steering_dyn_layer_all_layers/
#

import os
import subprocess
from pathlib import Path

TRAITS  = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS    = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
ALL_LAYERS = ",".join(map(str, range(32)))

# ==================== Calibration Job Template ====================
CALIB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dls_calib_all
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=log/dls_calib_all.out
#SBATCH --error=log/dls_calib_all.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
OUT_STATS="exp_steering_dyn_layer_all_layers/dls_calibration_stats_all.json"

mkdir -p "exp_steering_dyn_layer_all_layers"

echo "Starting Calibration for all 32 layers..."
"$PYTHON_BIN" scripts/04_dyn_layer/64_calibrate_dls_stats.py \\
    --config "$CONFIG" \\
    --vector_bank "$VECTOR_BANK" \\
    --prompts "$PROMPT_IN" \\
    --out_file "$OUT_STATS" \\
    --num_prompts 50 \\
    --layers "{layers}"
"""

# ==================== Evaluation Job Template ====================
EVAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dyn_all_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log/dyn_all_{trait}.out
#SBATCH --error=log/dyn_all_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

# 仮想環境のアクティベート
source persona_steering/bin/activate

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

OUT_DIR="exp_steering_dyn_layer_all_layers/results"
mkdir -p "$OUT_DIR"

CONFIG="config/mistral_7b.yaml"
VECTOR_BANK="exp_steering_layer_sweep/vectors/mean_diff_vectors.npz"
PROMPT_IN="exp_steering_layer_analysis/test_prompts_10.jsonl"
STATS="exp_steering_dyn_layer_all_layers/dls_calibration_stats_all.json"
JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

VALS=({vals_str})

for V in "${{VALS[@]}}"; do
    # ---------------- 1. Z-score Logit Diff ----------------
    echo "Running DLS Z-score logit_diff: Trait={trait}, Alpha=$V"
    JSONL_OUT="${{OUT_DIR}}/{trait}/logit_diff_Val${{V}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_logit_diff_Val${{V}}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/63_run_dyn_layer_zscore.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --stats_path "$STATS" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$V" \\
            --direction "high" \\
            --method "logit_diff" \\
            --layers "{layers}"
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

    # ---------------- 2. Z-score Anti Alignment ----------------
    echo "Running DLS Z-score anti_alignment: Trait={trait}, Alpha=$V"
    JSONL_OUT="${{OUT_DIR}}/{trait}/anti_alignment_Val${{V}}.jsonl"
    CSV_OUT="${{OUT_DIR}}/{trait}/scores_anti_alignment_Val${{V}}.csv"

    if [ ! -f "$JSONL_OUT" ]; then
        "$PYTHON_BIN" scripts/04_dyn_layer/63_run_dyn_layer_zscore.py \\
            --config "$CONFIG" \\
            --vector_bank "$VECTOR_BANK" \\
            --prompts "$PROMPT_IN" \\
            --stats_path "$STATS" \\
            --out_dir "$OUT_DIR" \\
            --axis "{trait}" \\
            --alpha "$V" \\
            --direction "high" \\
            --method "anti_alignment" \\
            --layers "{layers}"
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

    # ---------------- 3. Relative Anti Alignment ----------------
    echo "Running DLS Relative Anti-alignment: Trait={trait}, Alpha=$V"
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
            --direction "high" \\
            --layers "{layers}"
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
    job_dir = Path("jobs/dyn_layer_all")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. キャリブレーションジョブの生成と投入
    calib_content = CALIB_TEMPLATE.format(layers=ALL_LAYERS)
    calib_file = job_dir / "run_calibration_all.sh"
    with open(calib_file, "w") as f:
        f.write(calib_content)
    calib_file.chmod(0o755)

    print("Submitting calibration job...")
    res = subprocess.run(["sbatch", str(calib_file)], capture_output=True, text=True)
    stdout = res.stdout.strip()
    print(f"  {stdout}")
    
    # ジョブIDの抽出 ("Submitted batch job XXXXXX" の形式から XXXXXX を取得)
    if "batch job" not in stdout:
        print("[ERROR] Failed to submit calibration job.")
        return
    calib_job_id = stdout.split()[-1]

    # 2. 特性ごとの実験ジョブの生成と投入（キャリブレーション完了を待つ）
    vals_str = " ".join(map(str, VALS))
    for trait in TRAITS:
        eval_content = EVAL_TEMPLATE.format(
            trait=trait,
            vals_str=vals_str,
            layers=ALL_LAYERS,
        )
        eval_file = job_dir / f"run_all_{trait}.sh"
        with open(eval_file, "w") as f:
            f.write(eval_content)
        eval_file.chmod(0o755)

        print(f"Submitting eval job for {trait} (dependency on {calib_job_id})...")
        res = subprocess.run(
            ["sbatch", f"--dependency=afterok:{calib_job_id}", str(eval_file)],
            capture_output=True, text=True
        )
        print(f"  {res.stdout.strip()}")

if __name__ == "__main__":
    main()
