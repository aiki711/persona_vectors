#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_coordinate_step2.py
#

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res

def main():
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    out_base_dir = Path("exp_token_intensity/exp_dual_gating")
    out_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Fix theta_hi = 5.6 (the best trade-off from Step 1), sweep theta_lo
    t_hi = 5.6
    theta_lo_vals = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    k_lo = 8.0
    k_hi = 8.0
    
    results = []
    
    total_runs = len(theta_lo_vals)
    current_run = 0
    
    print(f"Starting Coordinate Descent Step 2: Fixing theta_IC = {t_hi}, Sweeping theta_H...")
    
    for t_lo in theta_lo_vals:
        current_run += 1
        config_name = f"Dual-{t_lo:.1f}-{t_hi}"
        print(f"\n[{current_run}/{total_runs}] Running Config: {config_name}")
        
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
                f"--gating_mode dual "
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
            csv_path = out_base_dir / trait / f"scores_masked_proj_rank_theta_{t_lo:.1f}_{t_hi}_k_{k_lo}_{k_hi}_dual_Val5.0.csv"
            if not csv_path.exists():
                csv_path_alt = out_base_dir / trait / f"scores_masked_proj_rank_theta_{float(t_lo)}_{t_hi}_k_{k_lo}_{k_hi}_dual_Val5.0.csv"
                if csv_path_alt.exists():
                    csv_path = csv_path_alt
                    
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
                "score": avg_score,
                "ppl": avg_ppl,
                "scores": trait_scores,
                "ppls": trait_ppls
            })
            print(f"--> Config {config_name} Summary: Score = {avg_score:.3f}, PPL = {avg_ppl:.3f}")
        else:
            print(f"--> Warning: Incomplete results for {config_name}")

    # Generate report
    report_path = out_base_dir / "coordinate_step2_summary.md"
    md_lines = [
        "# Coordinate Descent Optimization: Step 2 (Fixing theta_IC = 5.6)",
        "\nThis report presents the fine-grained tuning of the entropy threshold $\\theta_H$ to optimize syntax protection while keeping $\\theta_{IC} = 5.6$.\n",
        "## 1. Step 2 Sweep Results (theta_IC = 5.6)\n",
        "| Entropy Threshold (theta_H) | Alignment Score (Target: >4.34) | Perplexity (PPL) (Target: <10.46) | Result |",
        "| :---: | :---: | :---: | :--- |"
    ]
    
    best_config = None
    best_score_above_baseline = -1
    best_ppl = 999.0
    
    # Sort results by theta_lo
    results_sorted = sorted(results, key=lambda x: x["theta_lo"])
    
    for r in results_sorted:
        status = ""
        if r["score"] >= 4.34 and r["ppl"] < 10.46:
            status = "🏆 Dominated No Gating on BOTH!"
            if r["score"] > best_score_above_baseline or (r["score"] == best_score_above_baseline and r["ppl"] < best_ppl):
                best_score_above_baseline = r["score"]
                best_ppl = r["ppl"]
                best_config = r
        elif r["score"] >= 4.20:
            status = "⭐ Highly Efficient"
            
        md_lines.append(f"| **{r['theta_lo']:.1f}** | **{r['score']:.3f}** | **{r['ppl']:.3f}** | {status} |")
        
    if best_config:
        md_lines.append(f"\n### Recommended Absolute Optimal Config: **Dual-{best_config['theta_lo']:.1f}-5.6** (Score={best_config['score']:.3f}, PPL={best_config['ppl']:.3f})")
    else:
        best_tradeoff = sorted(results, key=lambda x: (-x["score"], x["ppl"]))[0]
        md_lines.append(f"\n### Recommended Trade-off Optimal Config: **Dual-{best_tradeoff['theta_lo']:.1f}-5.6** (Score={best_tradeoff['score']:.3f}, PPL={best_tradeoff['ppl']:.3f})")
        
    md_lines.append("\n## 2. Trait Breakdown Table (Scores / PPL)\n")
    md_lines.append("| Configuration | Extraversion | Neuroticism | Openness | Conscientiousness | Agreeableness |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results_sorted:
        scores = r["scores"]
        ppls = r["ppls"]
        md_lines.append(f"| **Dual-{r['theta_lo']:.1f}-5.6** | {scores[0]:.2f} / {ppls[0]:.1f} | {scores[1]:.2f} / {ppls[1]:.1f} | {scores[2]:.2f} / {ppls[2]:.1f} | {scores[3]:.2f} / {ppls[3]:.1f} | {scores[4]:.2f} / {ppls[4]:.1f} |")
        
    md_text = "\n".join(md_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\nSaved Step 2 report to: {report_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    shutil.copy(report_path, artifact_dir / "coordinate_step2_summary.md")
    print("Copied report to artifacts.")
    
    print("\n" + md_text)

if __name__ == "__main__":
    main()
