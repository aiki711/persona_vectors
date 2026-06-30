#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

# Input/Output paths
RESULTS_DIR = Path("exp_steering_dyn_layer_raw/results")
GENTIME_RESULTS_DIR = Path("exp_steering_dyn_gen_time_raw/results")
OUT_DIR = Path("exp_steering_dyn_layer_raw/figures/reduced_methods")
ARTIFACT_DIR = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion":      "Extraversion",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness":     "Agreeableness",
}

VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
ALPHAS_FIXED = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Focus strictly on Rank-based methods and baseline Logit-Diff
METHODS = [
    ("DLS Logit-Diff",        "logit_diff",             "#1abc9c"),  # Teal
    ("DLS Proj Rank-Only",    "proj_rank_only",         "#2ecc71"),  # Emerald Green
    ("PDF Rank-Only",         "masked_rank_only",       "#9b59b6"),  # Purple
    ("PDF Proj Rank-Only",    "masked_proj_rank_only",  "#e84393"),  # Pink / Magenta
]

GEN_TIME_METHODS = [
    ("Gen-Time Proj Rank-Only",    "proj_rank_only",         "#2ecc71"),
    ("Gen-Time PDF Rank-Only",     "masked_rank_only",       "#9b59b6"),
    ("Gen-Time PDF Proj Rank-Only", "masked_proj_rank_only",  "#e84393"),
]

# Formatting setup
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

def get_unsteered_baseline_score(results_dir: Path, trait: str) -> float:
    trait_dir = results_dir / trait
    for _, loader_key, _ in METHODS[1:]:
        for val in [1.0, 2.0, 4.0]:
            csv_path = trait_dir / f"scores_{loader_key}_Val{float(val)}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "base_score" in df.columns:
                        df["base_score"] = df["base_score"].replace(0, np.nan)
                        val_mean = df["base_score"].mean()
                        if not np.isnan(val_mean):
                            return val_mean
                except Exception:
                    pass
    return 3.0

def load_score_and_safety(results_dir: Path, trait: str, method: str, alpha: float) -> tuple[float, bool]:
    trait_dir = results_dir / trait
    csv_path = trait_dir / f"scores_{method}_Val{float(alpha)}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            score_col = "dyn_score" if "dyn_score" in df.columns else df.columns[2]
            df[score_col] = df[score_col].replace(0, np.nan)
            mean_score = df[score_col].mean()
            
            ppl_col = "dyn_ppl" if "dyn_ppl" in df.columns else "fusion_ppl"
            mean_ppl = df[ppl_col].mean()
            max_ppl = df[ppl_col].max()
            
            reason_col = "dyn_reason" if "dyn_reason" in df.columns else "fusion_reason"
            coherence_rate = df[reason_col].str.contains("Coherence: Yes", case=False, na=False).mean() if reason_col in df.columns else 1.0
            
            is_safe = (mean_ppl <= 25.0 and coherence_rate >= 0.8 and max_ppl <= 35.0)
            return mean_score, is_safe
        except Exception:
            pass
    return 0.0, False

# ----------------- 1. Fixed Alpha Prompt DLS plots -----------------
def plot_fixed_alpha_comparison():
    for alpha in ALPHAS_FIXED:
        data = []
        for trait in TRAITS:
            ub_score = get_unsteered_baseline_score(RESULTS_DIR, trait)
            method_results = {}
            for display_name, loader_key, _ in METHODS:
                score, is_safe = load_score_and_safety(RESULTS_DIR, trait, loader_key, alpha)
                method_results[display_name] = (score, is_safe)
            data.append({
                "trait": TRAIT_LABELS[trait],
                "Unsteered Baseline": (ub_score, True),
                **method_results
            })
            
        # Calculate averages
        avg_ub = np.mean([d["Unsteered Baseline"][0] for d in data])
        avg_results = {}
        for display_name, loader_key, _ in METHODS:
            scores = [d[display_name][0] for d in data if d[display_name][0] > 0.0]
            safeties = [d[display_name][1] for d in data if d[display_name][0] > 0.0]
            avg_score = np.mean(scores) if scores else 0.0
            is_all_safe = all(safeties) if safeties else False
            avg_results[display_name] = (avg_score, is_all_safe)
            
        data.append({
            "trait": "Average",
            "Unsteered Baseline": (avg_ub, True),
            **avg_results
        })
        
        categories = [d["trait"] for d in data]
        x = np.arange(len(categories))
        num_bars = 1 + len(METHODS)
        width = 0.14
        offset_start = - (num_bars - 1) / 2.0
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot Baseline
        ax.bar(x + (offset_start * width), [d["Unsteered Baseline"][0] for d in data], width, 
               label="Unsteered Baseline", color="#7f8c8d", zorder=3)
               
        # Plot methods
        for bar_idx, (display_name, _, color) in enumerate(METHODS):
            scores = []
            opacities = []
            hatches = []
            for d in data:
                score, is_safe = d[display_name]
                scores.append(score)
                if is_safe:
                    opacities.append(1.0)
                    hatches.append("")
                else:
                    opacities.append(0.4)
                    hatches.append("//")
                    
            x_offsets = x + ((offset_start + 1 + bar_idx) * width)
            for idx in range(len(data)):
                ax.bar(x_offsets[idx], scores[idx], width, 
                       color=color, alpha=opacities[idx], hatch=hatches[idx], edgecolor="black", 
                       linewidth=0.5 if hatches[idx] else 0.0, zorder=3,
                       label=display_name if idx == 0 else "")
                
                if scores[idx] > 0.0:
                    ax.annotate(f"{scores[idx]:.2f}",
                                 xy=(x_offsets[idx], scores[idx]),
                                 xytext=(0, 4),
                                 textcoords="offset points",
                                 ha="center", va="bottom",
                                 fontsize=8.5, fontweight="bold", color="#333333")
                    
        ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
        ax.set_title(f"Dynamic Steering Score Comparison (Alpha = {alpha})\n(Hatched Bars indicate unsafe configurations)", fontsize=15, fontweight="bold", pad=15)
        ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_ylim(0.8, 5.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9, ncol=2)
        
        file_name = f"score_compare_alpha_{alpha}.png"
        plt.savefig(OUT_DIR / file_name, dpi=200, bbox_inches="tight")
        shutil.copy(OUT_DIR / file_name, ARTIFACT_DIR / f"reduced_score_compare_alpha_{alpha}.png")
        plt.close()
        print(f"Generated comparison plot for alpha={alpha}")

# ----------------- 2. Gen-Time Alpha=5 comparison plot -----------------
def plot_gen_time_alpha_5():
    data = []
    for trait in TRAITS:
        ub_score = get_unsteered_baseline_score(GENTIME_RESULTS_DIR, trait)
        prompt_ld = load_score_and_safety(RESULTS_DIR, trait, "logit_diff", 5.0)[0]
        
        gen_results = {}
        for display_name, method_key, _ in GEN_TIME_METHODS:
            score = load_score_and_safety(GENTIME_RESULTS_DIR, trait, method_key, 5.0)[0]
            gen_results[display_name] = score
            
        data.append({
            "trait": TRAIT_LABELS[trait],
            "Unsteered Baseline": ub_score,
            "Prompt DLS Logit-Diff": prompt_ld,
            **gen_results
        })
        
    avg_row = {
        "trait": "Average",
        "Unsteered Baseline": np.mean([d["Unsteered Baseline"] for d in data]),
        "Prompt DLS Logit-Diff": np.mean([d["Prompt DLS Logit-Diff"] for d in data if not np.isnan(d["Prompt DLS Logit-Diff"])])
    }
    for display_name, _, _ in GEN_TIME_METHODS:
        avg_row[display_name] = np.mean([d[display_name] for d in data if not np.isnan(d[display_name])])
    data.append(avg_row)
    
    categories = [d["trait"] for d in data]
    x = np.arange(len(categories))
    num_bars = 5
    width = 0.14
    offset_start = - (num_bars - 1) / 2.0
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 1. Unsteered
    ax.bar(x + (offset_start * width), [d["Unsteered Baseline"] for d in data], width, label="Unsteered Baseline", color="#7f8c8d", zorder=3)
    # 2. Prompt Logit Diff
    ax.bar(x + ((offset_start + 1) * width), [d["Prompt DLS Logit-Diff"] for d in data], width, label="Prompt DLS Logit-Diff", color="#95a5a6", zorder=3)
    
    # 3. 3 Gen-Time methods
    for i, (display_name, _, color) in enumerate(GEN_TIME_METHODS):
        rects = ax.bar(x + ((offset_start + 2 + i) * width), [d[display_name] for d in data], width, label=display_name, color=color, zorder=3)
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            if h > 0.0 and not np.isnan(h):
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#333333")
                
    # Add labels on unsteered and prompt logit diff
    for idx, d in enumerate(data):
        ax.annotate(f"{d['Unsteered Baseline']:.2f}", xy=(x[idx] + (offset_start * width), d['Unsteered Baseline']), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{d['Prompt DLS Logit-Diff']:.2f}", xy=(x[idx] + ((offset_start + 1) * width), d['Prompt DLS Logit-Diff']), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_title("DLS Comparison at Alpha = 5.0 (Autoregressive Gen-Time DLS vs Prompt DLS)", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9, ncol=2)
    
    file_name = "gen_time_alpha_5_comparison.png"
    plt.savefig(OUT_DIR / file_name, dpi=200, bbox_inches="tight")
    shutil.copy(OUT_DIR / file_name, ARTIFACT_DIR / f"reduced_{file_name}")
    plt.close()
    print("Generated gen-time comparison plot.")

# ----------------- 3. Heatmaps and Max Safe Scores -----------------
def calculate_repetition_rate(text: str, n: int) -> float:
    if not isinstance(text, str):
        return 0.0
    words = [w.strip(".,!?:;()\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)

def load_summary(trait: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = RESULTS_DIR / trait
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        jsonl_path = trait_dir / f"{method}_Val{float(val)}.jsonl"
        if not csv_path.exists(): csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
        if not jsonl_path.exists(): jsonl_path = trait_dir / f"{method}_Val{val}.jsonl"
        
        if csv_path.exists() and jsonl_path.exists():
            try:
                df_csv = pd.read_csv(csv_path)
                dyn_texts = []
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if "dyn_text" in data:
                            dyn_texts.append(data["dyn_text"])
                rep_3gram = [calculate_repetition_rate(txt, 3) for txt in dyn_texts]
                rep_4gram = [calculate_repetition_rate(txt, 4) for txt in dyn_texts]
                
                coherence_rate = df_csv["dyn_reason"].str.contains("Coherence: Yes", case=False, na=False).mean() if "dyn_reason" in df_csv.columns else 1.0
                max_ppl = df_csv["dyn_ppl"].max() if "dyn_ppl" in df_csv.columns else np.nan
                safe_ppl_rate = (df_csv["dyn_ppl"] <= 25.0).mean() if "dyn_ppl" in df_csv.columns else 1.0
                
                records.append({
                    "val":       val,
                    "dyn_score": df_csv["dyn_score"].mean() if "dyn_score" in df_csv.columns else df_csv.iloc[:, 2].mean(),
                    "dyn_ppl":   df_csv["dyn_ppl"].mean() if "dyn_ppl" in df_csv.columns else np.nan,
                    "dyn_coherence_rate": coherence_rate,
                    "dyn_max_ppl": max_ppl,
                    "dyn_safe_ppl_rate": safe_ppl_rate,
                    "dyn_max_3gram_rep": max(rep_3gram) if rep_3gram else 0.0,
                    "dyn_max_4gram_rep": max(rep_4gram) if rep_4gram else 0.0,
                })
            except Exception as e:
                pass
    return pd.DataFrame(records)

def build_pivot(all_data):
    score_rows = {v: {} for v in VALS}
    ppl_rows = {v: {} for v in VALS}
    coherence_rows = {v: {} for v in VALS}
    safe_ppl_rate_rows = {v: {} for v in VALS}
    
    for display_name, loader_key, _ in METHODS:
        df = all_data.get(loader_key, pd.DataFrame())
        if df.empty: continue
        idx = df.set_index("val")
        for val in VALS:
            if val in idx.index:
                score_rows[val][display_name] = idx.loc[val, "dyn_score"]
                ppl_rows[val][display_name] = idx.loc[val, "dyn_ppl"]
                coherence_rows[val][display_name] = idx.loc[val, "dyn_coherence_rate"]
                safe_ppl_rate_rows[val][display_name] = idx.loc[val, "dyn_safe_ppl_rate"]
                
    p_score = pd.DataFrame.from_dict(score_rows, orient="index").reindex(VALS)
    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index").reindex(VALS)
    p_coherence = pd.DataFrame.from_dict(coherence_rows, orient="index").reindex(VALS)
    p_safe_ppl_rate = pd.DataFrame.from_dict(safe_ppl_rate_rows, orient="index").reindex(VALS)
    
    cols = [m[0] for m in METHODS if m[0] in p_score.columns]
    return p_score[cols], p_ppl[cols], p_coherence[cols], p_safe_ppl_rate[cols]

def plot_heatmap_and_summary():
    all_method_data = []
    
    for trait in TRAITS:
        method_data = {key: load_summary(trait, key) for _, key, _ in METHODS}
        all_method_data.append(method_data)
        
        p_score, p_ppl, p_coherence, p_safe_ppl_rate = build_pivot(method_data)
        
        # Plot individual heatmaps
        fig, ax = plt.subplots(figsize=(11, 10))
        sns.heatmap(p_score, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Steering Score"}, ax=ax, vmin=1.0, vmax=5.0)
        
        # Highlight safe cells (Mean PPL <= 20.0, Coherence >= 80%, Safe PPL rate >= 90%)
        for i in range(len(p_ppl.index)):
            for j in range(len(p_ppl.columns)):
                col_name = p_ppl.columns[j]
                val = p_ppl.index[i]
                if (p_ppl.iloc[i, j] <= 20.0 and p_coherence.loc[val, col_name] >= 0.8 and p_safe_ppl_rate.loc[val, col_name] >= 0.9):
                    rect = Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2.5, clip_on=False)
                    ax.add_patch(rect)
                    
        ax.set_title(f"Dynamic Steering Heatmap: {TRAIT_LABELS[trait]}\n(Black border indicates safe configuration)", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Steering Parameter Alpha (α)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dynamic Selection Methods", fontsize=11, fontweight="bold")
        
        file_name = f"heatmap_dyn_{trait}.png"
        plt.savefig(OUT_DIR / file_name, dpi=200, bbox_inches="tight")
        shutil.copy(OUT_DIR / file_name, ARTIFACT_DIR / f"reduced_{file_name}")
        plt.close()
        
    # Generate Summary Avg Heatmap
    p_score_avg = pd.DataFrame(0.0, index=VALS, columns=[m[0] for m in METHODS])
    p_ppl_avg = pd.DataFrame(0.0, index=VALS, columns=[m[0] for m in METHODS])
    p_coherence_avg = pd.DataFrame(0.0, index=VALS, columns=[m[0] for m in METHODS])
    p_safe_ppl_rate_avg = pd.DataFrame(0.0, index=VALS, columns=[m[0] for m in METHODS])
    
    for trait_idx, method_data in enumerate(all_method_data):
        ps, pp, pc, pr = build_pivot(method_data)
        p_score_avg += ps.fillna(0.0)
        p_ppl_avg += pp.fillna(999.0)
        p_coherence_avg += pc.fillna(0.0)
        p_safe_ppl_rate_avg += pr.fillna(0.0)
        
    p_score_avg /= len(TRAITS)
    p_ppl_avg /= len(TRAITS)
    p_coherence_avg /= len(TRAITS)
    p_safe_ppl_rate_avg /= len(TRAITS)
    
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(p_score_avg, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Avg Steering Score"}, ax=ax, vmin=1.0, vmax=5.0)
    
    for i in range(len(p_ppl_avg.index)):
        for j in range(len(p_ppl_avg.columns)):
            col_name = p_ppl_avg.columns[j]
            val = p_ppl_avg.index[i]
            # Average is safe only if it meets thresholds on average
            if (p_ppl_avg.iloc[i, j] <= 20.0 and p_coherence_avg.loc[val, col_name] >= 0.8 and p_safe_ppl_rate_avg.loc[val, col_name] >= 0.9):
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2.5, clip_on=False)
                ax.add_patch(rect)
                
    ax.set_title("DLS Summary Heatmap (Average of All 5 Traits)\n(Black border indicates safe configuration)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Steering Parameter Alpha (α)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dynamic Selection Methods", fontsize=11, fontweight="bold")
    
    file_name = "summary_dyn_all_traits.png"
    plt.savefig(OUT_DIR / file_name, dpi=200, bbox_inches="tight")
    shutil.copy(OUT_DIR / file_name, ARTIFACT_DIR / f"reduced_{file_name}")
    plt.close()
    print("Generated heatmaps and summary heatmap.")

def plot_max_safe_bar():
    # Load all and find max score under safety constraints
    bar_data = []
    for trait in TRAITS:
        trait_results = {}
        for display_name, loader_key, _ in METHODS:
            best_score = 0.0
            best_alpha = np.nan
            for alpha in VALS:
                score, is_safe = load_score_and_safety(RESULTS_DIR, trait, loader_key, alpha)
                if is_safe and score > best_score:
                    best_score = score
                    best_alpha = alpha
            trait_results[display_name] = (best_score, best_alpha)
        bar_data.append({
            "trait": TRAIT_LABELS[trait],
            **trait_results
        })
        
    # Calculate Average
    avg_results = {}
    for display_name, _, _ in METHODS:
        scores = [d[display_name][0] for d in bar_data if d[display_name][0] > 0.0]
        avg_results[display_name] = (np.mean(scores) if scores else 0.0, np.nan)
    bar_data.append({
        "trait": "Average",
        **avg_results
    })
    
    categories = [d["trait"] for d in bar_data]
    x = np.arange(len(categories))
    num_bars = len(METHODS)
    width = 0.14
    offset_start = - (num_bars - 1) / 2.0
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for bar_idx, (display_name, _, color) in enumerate(METHODS):
        scores = [d[display_name][0] for d in bar_data]
        alphas = [d[display_name][1] for d in bar_data]
        x_offsets = x + ((offset_start + bar_idx) * width)
        
        rects = ax.bar(x_offsets, scores, width, label=display_name, color=color, zorder=3)
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            if h > 0.0:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333")
                
                # Show alpha text inside the bar
                alpha_val = alphas[idx]
                if not np.isnan(alpha_val):
                    ax.annotate(f"α={alpha_val}",
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, -14), textcoords="offset points",
                                ha="center", va="top", fontsize=7, color="white", fontweight="bold", rotation=90)
                    
    ax.axhline(y=3.0, color="#cccccc", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_title("Maximum Practical Steering Score under Safety Criteria (Mean PPL <= 25.0, Max PPL <= 35.0, Coherence >= 80%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Steering Score (1.0 to 5.0)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.8, 5.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#bbbbbb", zorder=0)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.9, fontsize=9, ncol=2)
    
    file_name = "max_safe_score_compare.png"
    plt.savefig(OUT_DIR / file_name, dpi=200, bbox_inches="tight")
    shutil.copy(OUT_DIR / file_name, ARTIFACT_DIR / f"reduced_{file_name}")
    plt.close()
    print("Generated max safe score comparison bar chart.")

if __name__ == "__main__":
    print("Starting plotting for reduced set of methods (excluding all Cosine-based variants)...")
    plot_fixed_alpha_comparison()
    plot_gen_time_alpha_5()
    plot_heatmap_and_summary()
    plot_max_safe_bar()
    print("All plotting for reduced methods finished successfully!")
