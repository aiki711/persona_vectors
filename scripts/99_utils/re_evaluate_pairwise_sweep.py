import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path("exp_steering_layer_sweep_1-40/results") / args.trait
    out_dir = Path("exp_steering_layer_analysis/pairwise_results") / args.trait
    out_dir.mkdir(parents=True, exist_ok=True)

    python_bin = "/home/s2550009/persona_vectors/persona_steering/bin/python"
    eval_script = "scripts/02_base_steering/44_eval_pairwise_comparison.py"

    jsonl_files = sorted(list(results_dir.glob("layer_*.jsonl")))
    print(f"Found {len(jsonl_files)} files for trait: {args.trait}")

    for jsonl_path in tqdm(jsonl_files, desc=f"Pairwise Sweep [{args.trait}]"):
        csv_name = jsonl_path.name.replace(".jsonl", "_pairwise.csv")
        csv_path = out_dir / csv_name
        
        if csv_path.exists() and csv_path.stat().st_size > 0:
            continue

        cmd = [
            python_bin, eval_script,
            "--input", str(jsonl_path),
            "--output", str(csv_path),
            "--axis", args.trait
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[Error] Failed pairwise eval for {jsonl_path}")
            print(e.stderr)

if __name__ == "__main__":
    main()
