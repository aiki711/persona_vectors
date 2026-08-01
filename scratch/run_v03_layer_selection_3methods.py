#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_v03_layer_selection_3methods.py
# Run 3 Dynamic Layer Selection Methods (proj_rank, local_proj_rank, rank) on Mistral-7B-v0.3
# Alpha = 5.0 (Comparison with Logit-Diff)
# Output Directory: exp_layer_selection/exp_v03_layer_selection_comparison
#

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_layer_selection/exp_v03_layer_selection_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
METHODS = ["proj_rank", "local_proj_rank", "rank"]
ALPHA = "5.0"

def main():
    print("=======================================================")
    print("Starting 3 Dynamic Layer Selection Methods Evaluation on Mistral-7B-v0.3")
    print(f"Alpha: {ALPHA}")
    print(f"Output Directory: {OUT_DIR}")
    print("=======================================================")

    for mode in METHODS:
        print(f"\n==========================================")
        print(f"Running Layer Selection Method: {mode}")
        print(f"==========================================")

        for trait in TRAITS:
            trait_out_dir = OUT_DIR / trait
            trait_out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/01_layer_selection/82_run_dyn_layer_steering.py",
                "--config", "configs/mistral_7b.yaml",
                "--vector_bank", "vectors/mean_diff_vectors.npz",
                "--prompts", "inputs/eval_prompts_10.jsonl",
                "--mask_bank", "vectors/probe_masks.npz",
                "--out_dir", str(OUT_DIR),
                "--axis", trait,
                "--alpha", ALPHA,
                "--direction", "high",
                "--norm_mode", "raw_norm",
                "--score_mode", mode
            ]
            print(f"Generating for {trait} (mode={mode}, alpha={ALPHA})...")
            subprocess.run(cmd, cwd=WORKSPACE, check=True)

        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            print(f"Running LLM Judge eval for {trait} ({mode})...")
            subprocess.run(eval_cmd, cwd=WORKSPACE, check=True)

    print("\n-------------------------------------------------------")
    print("3 Dynamic Layer Selection Methods Evaluation Completed Successfully!")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()
