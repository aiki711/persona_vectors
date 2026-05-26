#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 86_plot_method_compare_heatmap.py
#
# Generates a methods-only comparison heatmap (no single-layer steering columns).
# Shows: DLS_logit_diff, DLS_anti_align, Fusion_Sigmoid, Fusion_Plateau, DLS_proj_prior
# Rows = alpha (Val) values; Columns = methods.
#
# Output: exp_steering_dyn_layer_proj_prior/figures/
#

import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

# Method definitions: (display_name, color, loader_key)
METHODS = [
    ("DLS_logit_diff",  "navy",       "logit_diff"),
    ("DLS_anti_align",  "darkred",    "anti_alignment"),
    ("Fusion_Sigmoid",  "darkorange", "sigmoid"),
    ("Fusion_Plateau",  "purple",     "soft_plateau"),
    ("DLS_proj_prior",  "darkcyan",   "proj_prior"),
]


# ── loaders ────────────────────────────────────────────────────────────────────

def load_dyn_summary(dyn_dir: Path, axis: str, method: str) -> pd.DataFrame:
    """Load DLS logit_diff / anti_alignment results (dyn_score, dyn_ppl columns)."""
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


def load_fusion_summary(fusion_dir: Path, axis: str, mode: str) -> pd.DataFrame:
    """Load Fusion (sigmoid / soft_plateau) results (dyn_score, dyn_ppl columns)."""
    records = []
    trait_dir = fusion_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_fusion_{mode}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_fusion_{mode}_Val{val}.csv"
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
    """Load Proj-Prior results evaluated by 62_eval_dyn_compare.py
    (dyn_score, dyn_ppl columns).
    """
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


def load_all_methods(all_layers_dir, fusion_dir, proj_prior_dir, axis):
    """Return dict of {method_key: DataFrame} for a single trait."""
    return {
        "logit_diff":    load_dyn_summary(all_layers_dir, axis, "logit_diff"),
        "anti_alignment": load_dyn_summary(all_layers_dir, axis, "anti_alignment"),
        "sigmoid":       load_fusion_summary(fusion_dir, axis, "sigmoid"),
        "soft_plateau":  load_fusion_summary(fusion_dir, axis, "soft_plateau"),
        "proj_prior":    load_proj_prior_summary(proj_prior_dir, axis),
    }


# ── build pivot tables ────────────────────────────────────────────────────────

def build_pivot(method_data_dict):
    """Build score and PPL pivot tables (rows=val, cols=method display name)."""
    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}

    for display_name, color, loader_key in METHODS:
        df = method_data_dict.get(loader_key, pd.DataFrame())
        if df.empty:
            continue
        idx = df.set_index("val")
        for val in VALS:
            if val in idx.index:
                score_rows[val][display_name] = idx.loc[val, "dyn_score"]
                ppl_rows[val][display_name]   = idx.loc[val, "dyn_ppl"]

    p_score = pd.DataFrame.from_dict(score_rows, orient="index")
    p_score.index.name = "val"
    p_score = p_score.reindex(VALS)
    p_score.columns = [m[0] for m in METHODS if m[0] in p_score.columns]

    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index")
    p_ppl.index.name = "val"
    p_ppl = p_ppl.reindex(VALS)
    p_ppl.columns = [m[0] for m in METHODS if m[0] in p_ppl.columns]

    return p_score, p_ppl


# ── helpers ────────────────────────────────────────────────────────────────────

def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            val = p_ppl.iloc[i, j]
            if not np.isnan(val) and val <= threshold:
                rect = Rectangle((j, i), 1, 1, fill=False,
                                  edgecolor="black", lw=2.5, clip_on=False)
                ax.add_patch(rect)


def draw_separators(ax, p_data):
    cols = list(p_data.columns)
    for display_name, color, _ in METHODS:
        if display_name in cols:
            ax.axvline(x=cols.index(display_name), color=color, linewidth=3.0)


# ── per-trait plot ─────────────────────────────────────────────────────────────

def plot_trait(axis, method_data_dict, out_dir, artifact_dir):
    print(f"\n[{axis}] plotting method-compare heatmap (no single-layer)...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_score, p_ppl = build_pivot(method_data_dict)

    # diagnostic
    pp_cols = list(p_score.columns) if not p_score.empty else []
    print(f"  Methods present: {pp_cols}")
    if "DLS_proj_prior" in pp_cols:
        safe_rows = p_ppl["DLS_proj_prior"][p_ppl["DLS_proj_prior"] <= 25.0]
        if not safe_rows.empty:
            best_alpha = p_score.loc[safe_rows.index, "DLS_proj_prior"].idxmax()
            best_score = p_score.loc[best_alpha, "DLS_proj_prior"]
            print(f"  DLS_proj_prior best safe alpha={best_alpha}, score={best_score:.3f}")

    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(8, n_methods * 1.5 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 12))

    configs = [
        (axes[0], p_score, p_ppl, f"Score [{axis.capitalize()}]",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   p_ppl, f"PPL   [{axis.capitalize()}]",
         "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty:
            ax_obj.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_obj.transAxes, fontsize=14)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 10})
        draw_separators(ax_obj, p_data)
        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title}"
            f" (Black Border: PPL ≤ 25.0 | Navy/Red: DLS | Orange/Purple: Fusion | Teal: Proj-Prior)",
            fontsize=11, fontweight="bold")
        ax_obj.set_xlabel("Method", fontsize=10)
        ax_obj.set_ylabel("Val (Steering Intensity / Alpha)", fontsize=10)

    plt.suptitle(
        f"Method Comparison (No Single-Layer Steering): {axis.capitalize()}",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = out_dir / f"method_compare_{axis}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir and artifact_dir.exists():
        dest = artifact_dir / f"method_compare_{axis}.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")


# ── summary (all traits avg) plot ─────────────────────────────────────────────

def plot_summary(all_method_data, out_dir, artifact_dir):
    """all_method_data: list of per-trait method_data_dict."""
    print("\n[Summary] plotting method-compare summary heatmap (all traits avg)...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate across traits
    score_acc = {k: {v: [] for v in VALS} for _, _, k in METHODS}
    ppl_acc   = {k: {v: [] for v in VALS} for _, _, k in METHODS}

    for method_data_dict in all_method_data:
        for _, _, loader_key in METHODS:
            df = method_data_dict.get(loader_key, pd.DataFrame())
            if df.empty:
                continue
            idx = df.set_index("val")
            for val in VALS:
                if val in idx.index:
                    score_acc[loader_key][val].append(idx.loc[val, "dyn_score"])
                    ppl_acc[loader_key][val].append(idx.loc[val, "dyn_ppl"])

    # Average
    score_rows = {v: {} for v in VALS}
    ppl_rows   = {v: {} for v in VALS}
    for display_name, _, loader_key in METHODS:
        for val in VALS:
            scores = score_acc[loader_key][val]
            ppls   = ppl_acc[loader_key][val]
            if scores:
                score_rows[val][display_name] = np.mean(scores)
                ppl_rows[val][display_name]   = np.mean(ppls)

    p_score = pd.DataFrame.from_dict(score_rows, orient="index")
    p_score.index.name = "val"
    p_score = p_score.reindex(VALS)

    p_ppl = pd.DataFrame.from_dict(ppl_rows, orient="index")
    p_ppl.index.name = "val"
    p_ppl = p_ppl.reindex(VALS)

    # Diagnostic
    print("  Methods in summary:", list(p_score.columns))
    if "DLS_proj_prior" in p_score.columns and "DLS_proj_prior" in p_ppl.columns:
        safe_mask = p_ppl["DLS_proj_prior"] <= 25.0
        if safe_mask.any():
            best_alpha = p_score.loc[safe_mask, "DLS_proj_prior"].idxmax()
            print(f"  DLS_proj_prior best safe: alpha={best_alpha}, "
                  f"score={p_score.loc[best_alpha, 'DLS_proj_prior']:.3f}, "
                  f"ppl={p_ppl.loc[best_alpha, 'DLS_proj_prior']:.2f}")
            safe_df = p_score[safe_mask]
            print(f"  DLS_proj_prior all-trait avg (safe PPL): "
                  f"{safe_df['DLS_proj_prior'].mean():.4f}")

    n_methods = len(p_score.columns) if not p_score.empty else 1
    fig_w = max(8, n_methods * 1.5 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 12))

    configs = [
        (axes[0], p_score, p_ppl, "Score (All Traits Avg)",
         "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   p_ppl, "PPL   (All Traits Avg)",
         "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        if p_data.empty:
            ax_obj.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_obj.transAxes, fontsize=14)
            continue
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.8, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 11})
        draw_separators(ax_obj, p_data)
        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title}"
            f" (Black Border: PPL ≤ 25.0 | Navy/Red: DLS | Orange/Purple: Fusion | Teal: Proj-Prior)",
            fontsize=11, fontweight="bold")
        ax_obj.set_xlabel("Method", fontsize=10)
        ax_obj.set_ylabel("Val (Steering Intensity / Alpha)", fontsize=10)

    plt.suptitle(
        "Method Comparison Summary (All Traits Avg, No Single-Layer Steering)",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = out_dir / "method_compare_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    if artifact_dir and artifact_dir.exists():
        dest = artifact_dir / "method_compare_summary.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot method-only comparison heatmaps (no single-layer steering).")
    ap.add_argument("--all_layers_dir", default="exp_steering_dyn_layer_all_layers_midpoint/results")
    ap.add_argument("--fusion_dir",     default="exp_steering_dyn_ic_fusion_midpoint/results")
    ap.add_argument("--proj_prior_dir", default="exp_steering_dyn_layer_proj_prior/results")
    ap.add_argument("--out_dir",        default="exp_steering_dyn_layer_proj_prior/figures")
    ap.add_argument("--artifact_dir",
                    default="/home/s2550009/.gemini/antigravity-ide/brain/"
                            "42af965e-7b98-48aa-bc1b-ea07d6f49983/images")
    args = ap.parse_args()

    all_layers_dir = Path(args.all_layers_dir)
    fusion_dir     = Path(args.fusion_dir)
    proj_prior_dir = Path(args.proj_prior_dir)
    out_dir        = Path(args.out_dir)
    artifact_dir   = Path(args.artifact_dir) if args.artifact_dir else None

    all_method_data = []

    for axis in TRAITS:
        method_data = load_all_methods(all_layers_dir, fusion_dir, proj_prior_dir, axis)
        all_method_data.append(method_data)
        plot_trait(axis, method_data, out_dir / axis, artifact_dir)

    plot_summary(all_method_data, out_dir, artifact_dir)
    print("\nMethod-compare heatmap generation finished.")


if __name__ == "__main__":
    main()
