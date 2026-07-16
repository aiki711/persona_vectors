#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_dense_dual_sweep.py
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
    
    # Define grid search space
    theta_lo_vals = [1.2, 1.5, 1.8]
    theta_hi_vals = [5.0, 5.5, 6.0, 6.5]
    k_lo = 8.0
    k_hi = 8.0
    
    results = []
    
    total_runs = len(theta_lo_vals) * len(theta_hi_vals)
    current_run = 0
    
    print(f"Starting Dense Dual Gating Grid Search ({total_runs} configurations)...")
    
    for t_lo in theta_lo_vals:
        for t_hi in theta_hi_vals:
            current_run += 1
            config_name = f"Dual-{t_lo}-{t_hi}"
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
                csv_path = out_base_dir / trait / f"scores_masked_proj_rank_theta_{t_lo}_{t_hi}_k_{k_lo}_{k_hi}_dual_Val5.0.csv"
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
                    "theta_hi": t_hi,
                    "score": avg_score,
                    "ppl": avg_ppl,
                    "scores": trait_scores,
                    "ppls": trait_ppls
                })
                print(f"--> Config {config_name} Summary: Score = {avg_score:.3f}, PPL = {avg_ppl:.3f}")
            else:
                print(f"--> Warning: Incomplete results for {config_name}")

    # Find Pareto optimal configurations
    pareto_configs = []
    for i, c1 in enumerate(results):
        dominated = False
        for j, c2 in enumerate(results):
            if i == j: continue
            # c2 dominates c1 if score2 >= score1 and ppl2 <= ppl1 with at least one inequality strict
            if (c2["score"] >= c1["score"] and c2["ppl"] <= c1["ppl"]) and (c2["score"] > c1["score"] or c2["ppl"] < c1["ppl"]):
                dominated = True
                break
        if not dominated:
            pareto_configs.append(c1)
            
    # Generate report
    report_path = out_base_dir / "dense_dual_grid_report.md"
    md_lines = [
        "# Dense Grid Sweep Report: Dual Gating Optimization",
        "\nThis report presents the results of a dense grid search over the dual gating parameters to identify the absolute sweet spot.\n",
        "## 1. Grid Performance Matrix\n",
        "| Entropy Threshold (theta_H) | Surprisal Threshold (theta_IC) | Alignment Score | Perplexity (PPL) | Status |",
        "| :---: | :---: | :---: | :---: | :--- |"
    ]
    
    # Sort results for readability
    results_sorted = sorted(results, key=lambda x: (x["theta_lo"], x["theta_hi"]))
    
    for r in results_sorted:
        is_pareto = any(p["theta_lo"] == r["theta_lo"] and p["theta_hi"] == r["theta_hi"] for p in pareto_configs)
        status = "🏆 Pareto Optimal" if is_pareto else ""
        if r["score"] >= 4.34 and r["ppl"] < 10.46:
            status += " 🌟 Dominates No Gating!"
            
        md_lines.append(f"| {r['theta_lo']} | {r['theta_hi']} | **{r['score']:.3f}** | **{r['ppl']:.3f}** | {status} |")
        
    md_lines.append("\n## 2. Trait Breakdown for Pareto-Optimal Configurations\n")
    md_lines.append("| Configuration (H / IC) | Extraversion | Neuroticism | Openness | Conscientiousness | Agreeableness |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for p in pareto_configs:
        scores = p["scores"]
        ppls = p["ppls"]
        md_lines.append(f"| **Dual-{p['theta_lo']}-{p['theta_hi']}** | {scores[0]:.2f} / {ppls[0]:.1f} | {scores[1]:.2f} / {ppls[1]:.1f} | {scores[2]:.2f} / {ppls[2]:.1f} | {scores[3]:.2f} / {ppls[3]:.1f} | {scores[4]:.2f} / {ppls[4]:.1f} |")
        
    md_text = "\n".join(md_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\nSaved report to: {report_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    shutil.copy(report_path, artifact_dir / "dense_dual_grid_report.md")
    print("Copied report to artifacts.")
    
    print("\n" + md_text)

if __name__ == "__main__":
    main()
