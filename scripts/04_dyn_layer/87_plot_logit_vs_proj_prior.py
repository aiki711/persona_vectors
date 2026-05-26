#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 87_plot_logit_vs_proj_prior.py
#
# Generates a side-by-side comparison heatmap of DLS_logit_diff vs DLS_proj_prior
# for all 5 personality traits in a single figure.
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
    "extraversion":     "Extraversion",
    "neuroticism":      "Neuroticism",
    "openness":         "Openness",
    "conscientiousness": "Conscientiousness",
    "agreeableness":    "Agreeableness",
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
    """Load Proj-Prior evaluated by 62_eval_dyn_compare.py (dyn_score, dyn_ppl)."""
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

def highlight_safe_cells(ax, ppl_array, threshold=25.0):
    """ppl_array: 2D numpy array (rows=vals, cols=methods)."""
    for i in range(ppl_array.shape[0]):
        for j in range(ppl_array.shape[1]):
            v = ppl_array[i, j]
            if not np.isnan(v) and v <= threshold:
                rect = Rectangle((j, i), 1, 1, fill=False,
                                  edgecolor="black", lw=2.5, clip_on=False)
                ax.add_patch(rect)


# ── main plot ──────────────────────────────────────────────────────────────────

def make_all_traits_figure(all_layers_dir, proj_prior_dir, out_dir, artifact_dir):
    """Build a single figure: 2 rows (Score / PPL) × N_traits columns.
    Each cell is a 2-column heatmap (Logit-Diff | Proj-Prior).
    """
    print("\n[All Traits] Building logit_diff vs proj_prior comparison figure...")

    n_traits = len(TRAITS)
    # --- collect data ---
    logit_data = {}   # trait -> (score_series, ppl_series)
    proj_data  = {}

    for axis in TRAITS:
        ld = load_dyn_summary(all_layers_dir, axis, "logit_diff")
        pp = load_proj_prior_summary(proj_prior_dir, axis)
        logit_data[axis] = (df_to_series(ld, "dyn_score"), df_to_series(ld, "dyn_ppl"))
        proj_data[axis]  = (df_to_series(pp, "dyn_score"), df_to_series(pp, "dyn_ppl"))

        # diagnostics
        if not pp.empty:
            ppl_s = df_to_series(pp, "dyn_ppl")
            safe_mask = ppl_s <= 25.0
            if safe_mask.any():
                sc_s = df_to_series(pp, "dyn_score")
                best_a = sc_s[safe_mask].idxmax()
                print(f"  [{axis}] proj_prior best safe alpha={best_a}, "
                      f"score={sc_s[best_a]:.3f}, ppl={ppl_s[best_a]:.2f}")

    METHOD_COLS = ["Logit-Diff", "Proj-Prior"]
    COL_COLORS  = ["navy", "darkcyan"]

    # figsize: ~3.5 per trait column, ~14 for 14 rows
    fig_w = n_traits * 4.0 + 1.5
    fig_h = 20
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        "DLS Logit-Diff vs Proj-Prior — All Personality Traits",
        fontsize=16, fontweight="bold", y=1.01)

    # 2 super-rows (Score / PPL), each subdivided into n_traits sub-columns
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)
    row_titles = ["Score", "PPL"]
    row_cmaps  = ["YlGn", "RdYlGn_r"]
    row_vmins  = [1, 1]
    row_vmaxs  = [5, 100]
    row_fmts   = [".2f", ".1f"]

    for row_idx, (row_title, cmap, vmin, vmax, fmt) in enumerate(
            zip(row_titles, row_cmaps, row_vmins, row_vmaxs, row_fmts)):

        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_traits, subplot_spec=outer[row_idx], wspace=0.3)

        for col_idx, axis in enumerate(TRAITS):
            ax = fig.add_subplot(inner[col_idx])

            if row_idx == 0:
                logit_s, logit_ppl = logit_data[axis]
                proj_s,  proj_ppl  = proj_data[axis]
                data_matrix = np.column_stack([logit_s.values, proj_s.values])
                ppl_matrix  = np.column_stack([logit_ppl.values, proj_ppl.values])
            else:
                logit_ppl = logit_data[axis][1]
                proj_ppl  = proj_data[axis][1]
                data_matrix = np.column_stack([logit_ppl.values, proj_ppl.values])
                ppl_matrix  = data_matrix

            plot_df = pd.DataFrame(
                data_matrix, index=VALS, columns=METHOD_COLS)
            plot_df.index.name = "alpha"

            sns.heatmap(
                plot_df, annot=True, fmt=fmt, cmap=cmap,
                vmin=vmin, vmax=vmax,
                linewidths=0.6, linecolor="gray",
                ax=ax, annot_kws={"size": 8},
                cbar=(col_idx == n_traits - 1),   # show colorbar only on last col
            )

            # Separator lines between the two methods
            for j, color in enumerate(COL_COLORS):
                ax.axvline(x=j, color=color, linewidth=2.5)

            # Black border for safe cells
            highlight_safe_cells(ax, ppl_matrix, threshold=25.0)

            trait_label = TRAIT_LABELS[axis]
            ax.set_title(
                f"{trait_label}\n({row_title})",
                fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Alpha (Val)", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

    # Add row super-labels on the left
    for row_idx, label in enumerate(row_titles):
        fig.text(0.005, 0.75 - row_idx * 0.5, label,
                 va="center", ha="left", fontsize=13, fontweight="bold",
                 rotation=90)

    # Legend note
    fig.text(0.5, -0.01,
             "Black border: PPL ≤ 25.0 (safe zone) | Navy: Logit-Diff | Teal: Proj-Prior",
             ha="center", fontsize=10, style="italic")

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
