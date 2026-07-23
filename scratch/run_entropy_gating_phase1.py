#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_entropy_gating_phase1.py
#

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def run_cmd(cmd):
    print(f"\nRunning command: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: Command failed with code {res.returncode}")
        print(f"Stdout:\n{res.stdout}")
        print(f"Stderr:\n{res.stderr}")
    else:
        print("Completed successfully.")
    return res

def main():
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    out_base_dir = Path("exp_token_intensity/exp_entropy_gating")
    out_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Sweep Grid for Phase 1 (Rise stage)
    theta_lo_vals = [1.2, 1.4, 1.6, 1.8]
    k_lo_vals = [1.5, 4.0, 8.0]
    
    results = []
    
    total_runs = len(theta_lo_vals) * len(k_lo_vals)
    current_run = 0
    
    print(f"Starting Rise-Stage Entropy Gating Sweep ({total_runs} configurations)...")
    
    for t_lo in theta_lo_vals:
        for k_lo in k_lo_vals:
            current_run += 1
            config_name = f"Entropy-Rise-{t_lo}-k-{k_lo}"
            print(f"\n[{current_run}/{total_runs}] Running Config: {config_name}")
            
            # Using dummy values for hi parameters as required by output file structure
            t_hi = 7.0
            k_hi = 2.0
            
            # 1. Run generation for all 5 traits
            for trait in traits:
                cmd_gen = (
                    f"python scripts/04_dyn_layer/02_token_intensity/run_token_intensity_steering.py "
                    f"--config configs/mistral_7b.yaml "
                    f"--vector_bank vectors/mean_diff_vectors.npz "
                    f"--prompts inputs/eval_prompts_10.jsonl "
                    f"--mask_bank vectors/soft_probe_masks.npz "
                    f"--out_dir {out_base_dir} "
                    f"--axis {trait} "
                    f"--alpha_max 5.0 "
                    f"--gating_mode entropy "
                    f"--static_layer "
                    f"--theta_lo {t_lo} --theta_hi {t_hi} "
                    f"--k_lo {k_lo} --k_hi {k_hi} "
                    f"--num_prompts 10"
                )
                run_cmd(cmd_gen)
                
            # 2. Run judge evaluation for all 5 traits
            for trait in traits:
                cmd_eval = (
                    f"python scripts/04_dyn_layer/02_token_intensity/batch_eval.py "
                    f"--results_dir {out_base_dir}/{trait} "
                    f"--axis {trait} "
                    f"--quant 4bit"
                )
                run_cmd(cmd_eval)
                
            # 3. Aggregate results
            trait_scores = []
            trait_ppls = []
            for trait in traits:
                csv_path = out_base_dir / trait / f"scores_masked_proj_rank_theta_{t_lo}_{t_hi}_k_{k_lo}_{k_hi}_entropy_Val5.0.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        trait_scores.append(df['dyn_score'].mean())
                        trait_ppls.append(df['dyn_ppl'].mean())
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")
                else:
                    print(f"Warning: CSV not found: {csv_path}")
                    
            if len(trait_scores) == 5:
                avg_score = np.mean(trait_scores)
                avg_ppl = np.mean(trait_ppls)
                results.append({
                    "theta_lo": t_lo,
                    "k_lo": k_lo,
                    "score": avg_score,
                    "ppl": avg_ppl,
                    "scores": trait_scores,
                    "ppls": trait_ppls
                })
                print(f"--> Config {config_name} Summary: Score = {avg_score:.3f}, PPL = {avg_ppl:.3f}")
            else:
                print(f"--> Warning: Incomplete results for {config_name}")

    # Generate Phase 1 Report
    report_path = out_base_dir / "entropy_gating_phase1_report.md"
    md_lines = [
        "# Rise-Stage Entropy Gating Sweep Report (Phase 1)",
        "\nThis report presents the sweep optimization results for the predictive entropy gate rise-stage parameters (theta_lo and k_lo).\n",
        "## 1. Performance Matrix (Ordered by Score)\n",
        "| Entropy Threshold (theta_lo) | Slope (k_lo) | Alignment Score | Perplexity (PPL) |",
        "| :---: | :---: | :---: | :---: |"
    ]
    
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    for r in results_sorted:
        md_lines.append(f"| {r['theta_lo']} | {r['k_lo']} | **{r['score']:.3f}** | **{r['ppl']:.3f}** |")
        
    md_text = "\n".join(md_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\nSaved Phase 1 report to: {report_path}")
    
    # Copy to artifacts
    try:
        artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/d66404fe-b75d-437e-af64-1fc20e801469")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(report_path, artifact_dir / "entropy_gating_phase1_report.md")
        print("Copied Phase 1 report to artifacts.")
    except Exception as e:
        print(f"Error copying to artifact: {e}")

if __name__ == "__main__":
    main()
