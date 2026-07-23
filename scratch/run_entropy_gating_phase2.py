#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_entropy_gating_phase2.py
# Phase 2: Fall-stage Entropy Gating Parameter Sweep Script
#

import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
LOG_DIR = WORKSPACE / "log"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

THETA_LO = 1.2
K_LO_LIST = [1.5, 4.0]
THETA_HI_LIST = [3.0, 4.5, 6.0]
K_HI_LIST = [1.0, 2.0]

def main():
    print("Starting Phase 2 (Fall-stage) Entropy Gating Sweep...")
    
    configs = []
    for k_lo in K_LO_LIST:
        for theta_hi in THETA_HI_LIST:
            for k_hi in K_HI_LIST:
                configs.append((THETA_LO, theta_hi, k_lo, k_hi))
                
    print(f"Total Configurations to Evaluate: {len(configs)}")
    
    for idx, (th_lo, th_hi, k_l, k_h) in enumerate(configs, 1):
        config_name = f"Entropy-Phase2-klo{k_l}-thi{th_hi}-khi{k_h}"
        # Check if all CSVs exist
        all_exist = True
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_l}_{k_h}_entropy_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                all_exist = False
                break
        if all_exist:
            print(f"Skipping Config {config_name}: All 5 trait CSVs already exist.")
            continue

        print(f"\n==========================================")
        print(f"[{idx}/{len(configs)}] Running Config: {config_name}")
        print(f"theta_lo={th_lo}, theta_hi={th_hi}, k_lo={k_l}, k_hi={k_h}")
        print(f"==========================================")
        for trait in TRAITS:
            cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py",
                "--config", "configs/mistral_7b.yaml",
                "--vector_bank", "vectors/mean_diff_vectors.npz",
                "--prompts", "inputs/eval_prompts_10.jsonl",
                "--mask_bank", "vectors/soft_probe_masks.npz",
                "--out_dir", str(OUT_DIR),
                "--axis", trait,
                "--alpha_max", "5.0",
                "--gating_mode", "entropy",
                "--static_layer",
                "--theta_lo", str(th_lo),
                "--theta_hi", str(th_hi),
                "--k_lo", str(k_l),
                "--k_hi", str(k_h),
                "--num_prompts", "10"
            ]
            res = subprocess.run(cmd, cwd=WORKSPACE)
            if res.returncode != 0:
                print(f"[ERROR] Steering generation failed for {trait} in {config_name}")
                
        # 2. Evaluation for each trait
        for trait in TRAITS:
            eval_cmd = [
                sys.executable, "-u", "scripts/04_dyn_layer/02_token_intensity/batch_eval.py",
                "--results_dir", str(OUT_DIR / trait),
                "--axis", trait,
                "--quant", "4bit"
            ]
            res_eval = subprocess.run(eval_cmd, cwd=WORKSPACE)
            if res_eval.returncode != 0:
                print(f"[ERROR] Evaluation failed for {trait} in {config_name}")

    # 3. Generate Summary Report
    print("\n------------------------------------------")
    print("Generating Phase 2 Summary Report...")
    print("------------------------------------------")
    
    summary_rows = []
    for (th_lo, th_hi, k_l, k_h) in configs:
        scores, ppls = [], []
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_l}_{k_h}_entropy_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
                
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    scores.append(df["dyn_score"].mean())
                    ppls.append(df["dyn_ppl"][np.isfinite(df["dyn_ppl"])].mean())
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    
        mean_score = np.mean(scores) if scores else 0.0
        mean_ppl = np.mean(ppls) if ppls else 999.0
        summary_rows.append({
            "theta_lo": th_lo,
            "k_lo": k_l,
            "theta_hi": th_hi,
            "k_hi": k_h,
            "score": mean_score,
            "ppl": mean_ppl
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by="score", ascending=False)
    
    report_path = OUT_DIR / "entropy_gating_phase2_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Fall-Stage Entropy Gating Sweep Report (Phase 2)\n\n")
        f.write("This report presents the sweep optimization results for the predictive entropy gate fall-stage parameters (theta_hi and k_hi).\n\n")
        f.write("## Performance Matrix (Ordered by Score)\n\n")
        f.write("| Rise Slope (k_lo) | Fall Threshold (theta_hi) | Fall Slope (k_hi) | Alignment Score | Perplexity (PPL) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for _, row in df_summary.iterrows():
            f.write(f"| {row['k_lo']} | {row['theta_hi']} | {row['k_hi']} | **{row['score']:.3f}** | **{row['ppl']:.3f}** |\n")
            
    print(f"Saved Phase 2 report to: {report_path}")
    if ARTIFACTS_DIR.exists():
        shutil.copy(report_path, ARTIFACTS_DIR / "entropy_gating_phase2_report.md")
        print("Copied Phase 2 report to artifacts.")
        
    print("\nPhase 2 Sweep completed successfully!")

if __name__ == "__main__":
    main()
