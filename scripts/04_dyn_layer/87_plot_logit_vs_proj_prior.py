#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 87_plot_logit_vs_proj_prior.py
#
# Generates a beautifully formatted side-by-side comparison heatmap of DLS_logit_diff vs DLS_proj_prior
# for all 5 personality traits.
#
# Output: exp_steering_dyn_layer_proj_prior/figures/logit_vs_proj_prior_all_traits.png
#

import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
TRAIT_LABELS = {
    "extraversion":      "Extraversion",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness":     "Agreeableness",
}
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


# ── loaders ────────────────────────────────────────────────────────────────────

def load_dyn_summary(dyn_dir: Path, axis: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = dyn_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                records.append({
                    "val":       val,
                    "dyn_score": df["dyn_score"].mean(),
                    "dyn_ppl":   df["dyn_ppl"].mean(),
                })
            except Exception:
                pass
    return pd.DataFrame(records)


def load_proj_prior_summary(proj_prior_dir: Path, axis: str) -> pd.DataFrame:
    """Load Proj-Prior evaluated results (dyn_score, dyn_ppl)."""
    records = []
    trait_dir = proj_prior_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_proj_prior_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_proj_prior_Val{val}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                records.append({
                    "val":       val,
                    "dyn_score": df["dyn_score"].mean(),
                    "dyn_ppl":   df["dyn_ppl"].mean(),
                })
            except Exception:
                pass
    return pd.DataFrame(records)


def df_to_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty:
        return pd.Series(np.nan, index=VALS)
    return df.set_index("val")[col].reindex(VALS)


# ── highlight ──────────────────────────────────────────────────────────────────

def highlight_safe_cells(ax, ppl_matrix, threshold=25.0):
    """
    Draws thick black borders around cells where ppl <= threshold.
    ppl_matrix shape: (2, 14) -> rows correspond to methods, cols to alphas.
    """
    for i in range(ppl_matrix.shape[0]):
        for j in range(ppl_matrix.shape[1]):
            v = ppl_matrix[i, j]
            if not np.isnan(v) and v <= threshold:
                # Rectangle(xy, width, height)
                # In seaborn heatmaps, columns are j (x-axis), rows are i (y-axis).
                # The cell is bounded by [j, j+1] and [i, i+1].
                rect = Rectangle((j, i), 1, 1, fill=False,
                                  edgecolor="black", lw=2.2, clip_on=False)
                ax.add_patch(rect)


# ── main plot ──────────────────────────────────────────────────────────────────

def make_all_traits_figure(all_layers_dir, proj_prior_dir, out_dir, artifact_dir):
    """
    Build a single figure: 5 rows (one per trait) x 2 columns (Score | PPL).
    Inside each subplot:
      - X-axis: Alpha (VALS)
      - Y-axis: Method (Logit-Diff, Proj-Prior)
    """
    print("\n[All Traits] Building logit_diff vs proj_prior comparison figure (Horizontal Layout)...")

    n_traits = len(TRAITS)
    # --- collect data ---
    logit_data = {}   # trait -> (score_series, ppl_series)
    proj_data  = {}

    for axis in TRAITS:
        ld = load_dyn_summary(all_layers_dir, axis, "logit_diff")
        pp = load_proj_prior_summary(proj_prior_dir, axis)
        logit_data[axis] = (df_to_series(ld, "dyn_score"), df_to_series(ld, "dyn_ppl"))
        proj_data[axis]  = (df_to_series(pp, "dyn_score"), df_to_series(pp, "dyn_ppl"))

        if not pp.empty:
            ppl_s = df_to_series(pp, "dyn_ppl")
            safe_mask = ppl_s <= 25.0
            if safe_mask.any():
                sc_s = df_to_series(pp, "dyn_score")
                best_a = sc_s[safe_mask].idxmax()
                print(f"  [{axis}] proj_prior best safe alpha={best_a}, "
                      f"score={sc_s[best_a]:.3f}, ppl={ppl_s[best_a]:.2f}")

    # Set up matplotlib style for clean aesthetic
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 16 inches wide, 9.5 inches tall
    fig = plt.figure(figsize=(16, 9.5))
    
    fig.suptitle(
        "DLS Method Comparison: Logit-Diff vs. Proposed (All Traits)",
        fontsize=18, fontweight="bold", y=0.98
    )
    fig.text(
        0.5, 0.93,
        "Comparison of dynamic layer steering methods across 14 scaling factors (alpha) for all 5 traits.\n"
        "Safe cells (Perplexity <= 25.0) are highlighted with bold black borders.",
        fontsize=11, style="italic", ha="center"
    )

    # GridSpec: 5 rows, 5 columns:
    #   Col 0: Score heatmap (width 1.0)
    #   Col 1: Score colorbar (width 0.02)
    #   Col 2: Spacer (width 0.12)
    #   Col 3: PPL heatmap (width 1.0)
    #   Col 4: PPL colorbar (width 0.02)
    gs = gridspec.GridSpec(
        5, 5, width_ratios=[1.0, 0.02, 0.12, 1.0, 0.02],
        wspace=0.05, hspace=0.45,
        left=0.12, right=0.95, top=0.86, bottom=0.08
    )

    # Add shared colorbars spanning all rows in Col 1 and Col 4
    cbar_score_ax = fig.add_subplot(gs[:, 1])
    cbar_ppl_ax = fig.add_subplot(gs[:, 4])

    METHOD_LABELS = ["Logit-Diff", "Proj-Prior"]

    for r, axis in enumerate(TRAITS):
        ax_score = fig.add_subplot(gs[r, 0])
        ax_ppl   = fig.add_subplot(gs[r, 3])

        # Load series
        logit_s, logit_ppl = logit_data[axis]
        proj_s,  proj_ppl  = proj_data[axis]

        # Construct (2, 14) matrices
        score_matrix = np.vstack([logit_s.values, proj_s.values])
        ppl_matrix   = np.vstack([logit_ppl.values, proj_ppl.values])

        # Convert to DataFrames
        df_score = pd.DataFrame(score_matrix, index=METHOD_LABELS, columns=VALS)
        df_ppl   = pd.DataFrame(ppl_matrix, index=METHOD_LABELS, columns=VALS)

        # ── 1. Plot Score Heatmap ──
        sns.heatmap(
            df_score, annot=True, fmt=".2f", cmap="YlGnBu",
            vmin=1.0, vmax=5.0, linewidths=0.5, linecolor="gainsboro",
            ax=ax_score, annot_kws={"size": 8.5, "weight": "semibold"},
            cbar=(r == 0), cbar_ax=cbar_score_ax if (r == 0) else None
        )
        highlight_safe_cells(ax_score, ppl_matrix, threshold=25.0)

        # Style Score Subplot
        ax_score.set_ylabel(TRAIT_LABELS[axis], fontsize=12, fontweight="bold", labelpad=15)
        if r == 0:
            ax_score.set_title("Steering Score (1.0 to 5.0, Higher is Better)", fontsize=11, fontweight="bold", pad=10)
        
        # Hide tick labels/labels based on position
        if r < 4:
            ax_score.set_xticklabels([])
        else:
            ax_score.set_xlabel("Alpha (Val)", fontsize=10)
            ax_score.set_xticklabels(VALS, fontsize=9)
            
        ax_score.set_yticklabels(METHOD_LABELS, rotation=0, fontsize=9)

        # ── 2. Plot PPL Heatmap ──
        sns.heatmap(
            df_ppl, annot=True, fmt=".1f", cmap="YlOrRd",
            vmin=5.0, vmax=50.0, linewidths=0.5, linecolor="gainsboro",
            ax=ax_ppl, annot_kws={"size": 8.5, "weight": "semibold"},
            cbar=(r == 0), cbar_ax=cbar_ppl_ax if (r == 0) else None
        )
        highlight_safe_cells(ax_ppl, ppl_matrix, threshold=25.0)

        # Style PPL Subplot
        if r == 0:
            ax_ppl.set_title("Perplexity (PPL, Lower is Better, Safe <= 25.0)", fontsize=11, fontweight="bold", pad=10)
        
        # Hide labels based on position
        if r < 4:
            ax_ppl.set_xticklabels([])
        else:
            ax_ppl.set_xlabel("Alpha (Val)", fontsize=10)
            ax_ppl.set_xticklabels(VALS, fontsize=9)
            
        ax_ppl.set_yticklabels([]) # Hide y-ticks for PPL since it's aligned with Score
        ax_ppl.set_ylabel("")

    # Adjust colorbar labels/titles
    cbar_score_ax.set_ylabel("Score Scale", fontsize=10, fontweight="bold")
    cbar_ppl_ax.set_ylabel("PPL Scale (Clipped at 50.0)", fontsize=10, fontweight="bold")

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logit_vs_proj_prior_all_traits.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir and artifact_dir.exists():
        dest = artifact_dir / "logit_vs_proj_prior_all_traits.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Logit-Diff vs Proj-Prior comparison across all traits.")
    ap.add_argument("--all_layers_dir", default="exp_steering_dyn_layer_all_layers_midpoint/results")
    ap.add_argument("--proj_prior_dir", default="exp_steering_dyn_layer_proj_prior/results")
    ap.add_argument("--out_dir",        default="exp_steering_dyn_layer_proj_prior/figures")
    ap.add_argument("--artifact_dir",
                    default="/home/s2550009/.gemini/antigravity-ide/brain/"
                            "42af965e-7b98-48aa-bc1b-ea07d6f49983/images")
    args = ap.parse_args()

    all_layers_dir = Path(args.all_layers_dir)
    proj_prior_dir = Path(args.proj_prior_dir)
    out_dir        = Path(args.out_dir)
    artifact_dir   = Path(args.artifact_dir) if args.artifact_dir else None

    make_all_traits_figure(all_layers_dir, proj_prior_dir, out_dir, artifact_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
