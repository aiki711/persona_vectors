import os
import glob
import subprocess
import argparse
import re
from pathlib import Path
from tqdm import tqdm

def get_line_count(file_path: Path) -> int:
    """JSONLファイルの行数（プロンプト数）をカウントする"""
    if not file_path.exists():
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, default=None, help="Specific trait to evaluate (e.g. extraversion)")
    parser.add_argument("--force", action="store_true", help="Overwrite everything")
    parser.add_argument("--expected_lines", type=int, default=10, help="Expected number of prompts per file")
    args = parser.parse_args()

    base_dir = Path("exp_steering_layer_sweep_1-40/results")
    all_traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    
    if args.trait:
        if args.trait not in all_traits:
            print(f"[Error] Unknown trait: {args.trait}")
            return
        traits = [args.trait]
    else:
        traits = all_traits

    python_bin = "/home/s2550009/persona_vectors/persona_steering/bin/python"
    eval_script = "scripts/33_eval_adaptive_steering.py"
    run_script = "scripts/40_run_layer_sweep.py"
    judge_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    config = "config/mistral_7b.yaml"
    vector_bank = "exp_steering_layer_sweep_1-40/vectors/mean_diff_vectors.npz"
    prompts = "exp_steering_layer_sweep_1-40/test_prompts_10.jsonl"

    # リストアップ
    jsonl_files = []
    for trait in traits:
        trait_dir = base_dir / trait
        if not trait_dir.exists():
            continue
        jsonl_files.extend(list(trait_dir.glob("layer_*.jsonl")))

    print(f"Checking {len(jsonl_files)} files for integrity and evaluation status...")

    for jsonl_path in tqdm(jsonl_files, desc=f"Processing {args.trait or 'all'}"):
        trait = jsonl_path.parent.name
        # 最初から正しい名前 (scores_layer_...) で出力するように変更
        csv_name = jsonl_path.name.replace("layer_", "scores_layer_").replace(".jsonl", ".csv")
        csv_path = jsonl_path.parent / csv_name
        
        # 1. 整合性チェック: 生成ファイルが完全か確認
        line_count = get_line_count(jsonl_path)
        is_incomplete = (line_count < args.expected_lines)
        
        if is_incomplete:
            print(f"\n[Integrity] Incomplete generation found: {jsonl_path} ({line_count}/{args.expected_lines} lines). Regenerating...")
            # layer と alpha を抽出
            match = re.search(r"layer_(\d+)_Val([\d\.]+)", jsonl_path.name)
            if match:
                layer_id = match.group(1)
                alpha_val = match.group(2)
                
                # 再生成コマンドの実行
                run_cmd = [
                    python_bin, run_script,
                    "--config", config,
                    "--vector_bank", vector_bank,
                    "--prompts", prompts,
                    "--out_dir", str(jsonl_path.parent),
                    "--axis", trait,
                    "--target_layer", layer_id,
                    "--tau", alpha_val,
                    "--alpha", alpha_val,
                    "--mode", "both"
                ]
                subprocess.run(run_cmd, check=True)
                # 再生成されたので line_count を更新
                line_count = get_line_count(jsonl_path)
            else:
                print(f"[Error] Could not parse layer/alpha from filename: {jsonl_path.name}")
                continue

        # 2. 評価済みチェック
        if not args.force and csv_path.exists() and csv_path.stat().st_size > 0:
            # 既存のCSVの行数もチェック（ヘッダ1行 + データ10行 = 11行）
            csv_lines = get_line_count(csv_path)
            if csv_lines >= args.expected_lines + 1:
                continue

        # 3. 評価コマンドの実行
        cmd = [
            python_bin, eval_script,
            "--input", str(jsonl_path),
            "--output", str(csv_path),
            "--axis", trait,
            "--model", judge_model
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[Error] Failed to evaluate {jsonl_path}")
            print(e.stderr)

    print(f"\nCompleted evaluation for traits: {traits}")

if __name__ == "__main__":
    main()
