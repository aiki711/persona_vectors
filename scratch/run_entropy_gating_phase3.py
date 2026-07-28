#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_entropy_gating_phase3.py
# Phase 3: Fine-grained & Extended Range Fall-stage Entropy Gating Parameter Sweep
#

import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE = Path("/home/s2550009/persona_vectors")
OUT_DIR = WORKSPACE / "exp_token_intensity/exp_entropy_gating"
LOG_DIR = WORKSPACE / "log"
ARTIFACTS_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

THETA_LO = 1.2
K_LO = 1.5

THETA_HI_LIST = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
K_HI_LIST = [0.5, 1.0, 1.5, 2.0]

def main():
    print("Starting Phase 3 (Fine-grained & Extended Range) Entropy Gating Sweep...")
    
    configs = []
    for theta_hi in THETA_HI_LIST:
        for k_hi in K_HI_LIST:
            configs.append((THETA_LO, theta_hi, K_LO, k_hi))
                
    print(f"Total Configurations to Evaluate: {len(configs)}")
    
    for idx, (th_lo, th_hi, k_l, k_h) in enumerate(configs, 1):
        config_name = f"Entropy-Phase3-klo{k_l}-thi{th_hi}-khi{k_h}"
        # Check if all CSVs exist
        all_exist = True
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_l}_{k_h}_entropy_plateau_Val5.0.csv"
                csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                all_exist = False
                break
        if all_exist:
            print(f"Skipping Config [{idx}/{len(configs)}] {config_name}: All 5 trait CSVs already exist.")
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
                "--gating_mode", "entropy_plateau",
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
    print("Generating Phase 3 Summary Report...")
    print("------------------------------------------")
    
    summary_rows = []
    for (th_lo, th_hi, k_l, k_h) in configs:
        scores, ppls = [], []
        for trait in TRAITS:
            csv_name = f"scores_masked_proj_rank_theta_{th_lo:.1f}_{th_hi:.1f}_k_{k_l:.1f}_{k_h:.1f}_entropy_plateau_Val5.0.csv"
            csv_path = OUT_DIR / trait / csv_name
            if not csv_path.exists():
                csv_name = f"scores_masked_proj_rank_theta_{th_lo}_{th_hi}_k_{k_l}_{k_h}_entropy_plateau_Val5.0.csv"
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
    
    report_path = OUT_DIR / "entropy_gating_phase3_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 (Fine-Grained & Extended) Entropy Gating Sweep Report\n\n")
        f.write("This report presents the fine-grained grid sweep optimization results for predictive entropy fall-stage parameters (theta_hi: 4.0 to 9.0, k_hi: 0.5 to 2.0).\n\n")
        f.write("## Overall Ranking (Ordered by Mean Alignment Score)\n\n")
        f.write("| Rank | Fall Threshold (theta_hi) | Fall Slope (k_hi) | Mean Alignment Score | Mean Perplexity (PPL) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for rank, (_, row) in enumerate(df_summary.iterrows(), 1):
            f.write(f"| {rank} | {row['theta_hi']} | {row['k_hi']} | **{row['score']:.3f}** | **{row['ppl']:.3f}** |\n")

    print(f"Saved Phase 3 report to: {report_path}")
    
    # Run heatmap plotting script
    try:
        subprocess.run([sys.executable, "scratch/plot_entropy_gating_phase3_heatmaps.py"], cwd=WORKSPACE)
    except Exception as e:
        print(f"Failed to plot Phase 3 heatmaps: {e}")

if __name__ == "__main__":
    main()
