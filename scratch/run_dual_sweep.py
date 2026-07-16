#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scratch/run_dual_sweep.py
#

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
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
    out_base_dir = Path("exp_token_intensity/exp_dual_gating")
    out_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Dual configurations (sharp logic for both boundaries: k_lo = 8.0, k_hi = 8.0)
    # theta_lo is the entropy syntax threshold, theta_hi is the IC rare-word threshold
    configs = [
        {"name": "Dual-Sharp-1.5-6.0", "theta_lo": 1.5, "theta_hi": 6.0, "k_lo": 8.0, "k_hi": 8.0},
        {"name": "Dual-Sharp-1.8-6.0", "theta_lo": 1.8, "theta_hi": 6.0, "k_lo": 8.0, "k_hi": 8.0},
        {"name": "Dual-Sharp-1.5-5.5", "theta_lo": 1.5, "theta_hi": 5.5, "k_lo": 8.0, "k_hi": 8.0}
    ]
    
    sweep_results = []
    
    for cfg in configs:
        name = cfg["name"]
        t_lo = cfg["theta_lo"]
        t_hi = cfg["theta_hi"]
        k_lo = cfg["k_lo"]
        k_hi = cfg["k_hi"]
        
        print(f"\n==================================================================")
        print(f"Starting Sweep for Config: {name}")
        print(f"==================================================================")
        
        # 1. Run generation
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
            
        # 2. Run judge evaluation
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
            sweep_results.append({
                "Config": name,
                "Theta_H": t_lo,
                "Theta_IC": t_hi,
                "Score": avg_score,
                "PPL": avg_ppl,
                "Trait_Scores": trait_scores,
                "Trait_PPLs": trait_ppls
            })
            print(f"Config {name} Summary: Score = {avg_score:.3f}, PPL = {avg_ppl:.3f}")
        else:
            print(f"Warning: Incomplete results for {name}")

    # Write summary report
    summary_path = out_base_dir / "dual_gating_summary.md"
    md_lines = [
        "# Dual Gating (Predictive Entropy + Surprisal Cliff) Experiment Results",
        "\nThis report evaluates the **Dual Gating** method which combines zero-delay entropy gating (syntax protection) "
        "with offline surprisal cliff gating (rare word protection) at $\\alpha_{max}=5.0$.\n",
        "## 1. Overall Performance Table\n",
        "| Configuration | Entropy Thresh (theta_H) | IC Thresh (theta_IC) | Alignment Score (Target: >4.34) | Perplexity (PPL) (Target: <10.46) | Result |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for res in sweep_results:
        status = "❌ Failed to dominate"
        if res["Score"] >= 4.34 and res["PPL"] < 10.46:
            status = "🏆 Dominated No Gating on BOTH!"
        elif res["Score"] >= 4.20 and res["PPL"] < 10.0:
            status = "⭐ Near Dominance (Highly Efficient)"
            
        md_lines.append(f"| **{res['Config']}** | {res['Theta_H']} | {res['Theta_IC']} | **{res['Score']:.3f}** | **{res['PPL']:.3f}** | {status} |")
    
    md_lines.append("\n## 2. Trait Breakdown Table (Scores / PPL)\n")
    md_lines.append("| Configuration | Extraversion | Neuroticism | Openness | Conscientiousness | Agreeableness |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for res in sweep_results:
        scores = res["Trait_Scores"]
        ppls = res["Trait_PPLs"]
        md_lines.append(f"| **{res['Config']}** | {scores[0]:.2f} / {ppls[0]:.1f} | {scores[1]:.2f} / {ppls[1]:.1f} | {scores[2]:.2f} / {ppls[2]:.1f} | {scores[3]:.2f} / {ppls[3]:.1f} | {scores[4]:.2f} / {ppls[4]:.1f} |")
        
    md_text = "\n".join(md_lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\nSaved dual gating report to: {summary_path}")
    
    # Copy to artifacts
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    shutil.copy(summary_path, artifact_dir / "dual_gating_summary.md")
    print("Copied report to artifacts.")
    
    print("\n" + md_text)

if __name__ == "__main__":
    main()
