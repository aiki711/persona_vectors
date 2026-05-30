#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 84_plot_proj_prior_heatmap.py
#
# Generates unified 2x1 heatmaps comparing all methods including the new
# Projection & Prior DLS method.
#
# Saves output to a dedicated folder and copies with distinct names to artifacts.
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
LAYERS = list(range(32))
VALS   = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def load_summary(input_dir: Path, axis: str) -> pd.DataFrame:
    records = []
    trait_dir = input_dir / axis
    for layer in LAYERS:
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{layer}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{layer}_Val{val}.csv"
                if not csv_path.exists():
                    continue
            try:
                df = pd.read_csv(csv_path)
                records.append({
                    "layer": layer,
                    "val":   val,
                    "const_score": df["const_score"].mean(),
                    "const_ppl":   df["const_ppl"].mean(),
                })
            except Exception:
                pass
    return pd.DataFrame(records)

def load_dyn_summary(dyn_dir: Path, axis: str, method: str) -> pd.DataFrame:
    records = []
    trait_dir = dyn_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_{method}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_{method}_Val{val}.csv"
            if not csv_path.exists():
                continue
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
    records = []
    trait_dir = fusion_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_fusion_{mode}_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_fusion_{mode}_Val{val}.csv"
            if not csv_path.exists():
                continue
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
    """Load proj-prior evaluated results.
    62_eval_dyn_compare.py writes: dyn_score, dyn_ppl, base_score, base_ppl, ...
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

def load_cos_prior_summary(proj_prior_dir: Path, axis: str) -> pd.DataFrame:
    """Load cos-prior evaluated results."""
    records = []
    trait_dir = proj_prior_dir / axis
    for val in VALS:
        csv_path = trait_dir / f"scores_cos_prior_Val{float(val)}.csv"
        if not csv_path.exists():
            csv_path = trait_dir / f"scores_cos_prior_Val{val}.csv"
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

def highlight_safe_cells(ax, p_ppl, threshold=25.0):
    if p_ppl is None or p_ppl.empty:
        return
    for i in range(len(p_ppl.index)):
        for j in range(len(p_ppl.columns)):
            val = p_ppl.iloc[i, j]
            if not np.isnan(val) and val <= threshold:
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2.5, clip_on=False)
                ax.add_patch(rect)

def make_empty_pivot(vals):
    return pd.DataFrame(index=pd.Index(vals, name="val"))

def append_comparison_cols(pivot_df, score_col, logit_df, anti_df, sig_df, plat_df, proj_prior_df, cos_prior_df):
    result = pivot_df.copy()
    if not logit_df.empty and score_col in logit_df.columns:
        result["DLS_logit_diff"] = logit_df.set_index("val")[score_col]
    if not anti_df.empty and score_col in anti_df.columns:
        result["DLS_anti_align"] = anti_df.set_index("val")[score_col]
    if not sig_df.empty and score_col in sig_df.columns:
        result["Fusion_Sigmoid"] = sig_df.set_index("val")[score_col]
    if not plat_df.empty and score_col in plat_df.columns:
        result["Fusion_Plateau"] = plat_df.set_index("val")[score_col]
    if not proj_prior_df.empty and score_col in proj_prior_df.columns:
        result["DLS_proj_prior"] = proj_prior_df.set_index("val")[score_col]
    if not cos_prior_df.empty and score_col in cos_prior_df.columns:
        result["DLS_cos_prior"] = cos_prior_df.set_index("val")[score_col]
    return result

def plot_axis(df: pd.DataFrame, axis: str, out_dir: Path, all_layers_dir: Path, fusion_dir: Path, proj_prior_dir: Path, artifact_dir: Path | None):
    print(f"\n[{axis}] plotting unified full-layer projection-prior heatmap...")
    plt.close("all")
    out_dir.mkdir(parents=True, exist_ok=True)

    logit_df      = load_dyn_summary(all_layers_dir, axis, "logit_diff")
    anti_df       = load_dyn_summary(all_layers_dir, axis, "anti_alignment")
    sig_df        = load_fusion_summary(fusion_dir, axis, "sigmoid")
    plat_df       = load_fusion_summary(fusion_dir, axis, "soft_plateau")
    proj_prior_df = load_proj_prior_summary(proj_prior_dir, axis)
    cos_prior_df  = load_cos_prior_summary(proj_prior_dir, axis)

    has_layer_data = not df.empty
    if has_layer_data:
        def pivot(col):
            return df.pivot(index="val", columns="layer", values=col)
        p_score = pivot("const_score")
        p_ppl   = pivot("const_ppl")
    else:
        p_score = make_empty_pivot(VALS)
        p_ppl   = make_empty_pivot(VALS)

    p_score_comp = append_comparison_cols(p_score, "dyn_score", logit_df, anti_df, sig_df, plat_df, proj_prior_df, cos_prior_df)
    p_ppl_comp   = append_comparison_cols(p_ppl,   "dyn_ppl",   logit_df, anti_df, sig_df, plat_df, proj_prior_df, cos_prior_df)

    p_score_comp = p_score_comp.reindex(VALS)
    p_ppl_comp   = p_ppl_comp.reindex(VALS)

    fig, axes = plt.subplots(2, 1, figsize=(28, 16))
    configs = [
        (axes[0], p_score_comp, p_ppl_comp, "Single-Layer Steering — Score", "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl_comp,   p_ppl_comp, "Single-Layer Steering — PPL",   "RdYlGn_r", 1, 100, ".1f"),
    ]

    separators = [
        ("DLS_logit_diff", "navy"),
        ("DLS_anti_align", "darkred"),
        ("Fusion_Sigmoid", "darkorange"),
        ("Fusion_Plateau", "purple"),
        ("DLS_proj_prior", "darkcyan"),
        ("DLS_cos_prior", "coral"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})

        cols = list(p_data.columns)
        for col_name, color in separators:
            if col_name in cols:
                ax_obj.axvline(x=cols.index(col_name), color=color, linewidth=3.0)

        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title} [{axis.capitalize()}]"
            f" (Black Border: PPL <= 25.0 | Navy/Red: DLS | Orange/Purple: Raw-Norm Fusion | Teal: Proj-Prior)",
            fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer (0 to 31) / Evaluation Variants (rightmost)", fontsize=10)
        ax_obj.set_ylabel("Val (Steering Intensity)", fontsize=10)

    plt.suptitle(f"Unified 32-Layer Steering & DLS/Fusion Comparison (with Proj-Prior DLS): {axis.capitalize()}",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{axis}_proj_prior_compare.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap to: {out_path}")

    # Copy to artifact folder if available
    if artifact_dir and artifact_dir.exists():
        dest = artifact_dir / f"heatmap_{axis}_proj_prior_compare.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")

    # Plot single-layer only heatmap with reversed y-axis
    if has_layer_data:
        print(f"  Plotting single-layer only heatmap for {axis} (reversed y-axis)...")
        p_score_rev = p_score.reindex(reversed(VALS))
        p_ppl_rev   = p_ppl.reindex(reversed(VALS))

        fig_s, axes_s = plt.subplots(2, 1, figsize=(22, 14))
        configs_s = [
            (axes_s[0], p_score_rev, p_ppl_rev, "Single-Layer Steering (Only) — Score", "YlGn",     1,   5, ".2f"),
            (axes_s[1], p_ppl_rev,   p_ppl_rev, "Single-Layer Steering (Only) — PPL",   "RdYlGn_r", 1, 100, ".1f"),
        ]

        for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs_s:
            sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        linewidths=0.4, linecolor="gray",
                        ax=ax_obj, annot_kws={"size": 8})

            highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
            ax_obj.set_title(
                f"{title} [{axis.capitalize()}] (Black Border: PPL <= 25.0)",
                fontsize=12, fontweight="bold")
            ax_obj.set_xlabel("Layer (0 to 31)", fontsize=10)
            ax_obj.set_ylabel("Val (Steering Intensity)", fontsize=10)

        plt.suptitle(f"Single-Layer Steering Sweep (Only): {axis.capitalize()} (Reversed Steering Intensity Axis)",
                     fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()

        out_path_s = out_dir / f"heatmap_{axis}_single_layer_only.png"
        plt.savefig(out_path_s, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    Saved single-layer only heatmap to: {out_path_s}")

        if artifact_dir and artifact_dir.exists():
            dest_s = artifact_dir / f"heatmap_{axis}_single_layer_only.png"
            shutil.copy(out_path_s, dest_s)
            print(f"    Copied to artifact: {dest_s}")

def make_summary_heatmaps(all_df: pd.DataFrame,
                          logit_all_df: pd.DataFrame,
                          anti_all_df: pd.DataFrame,
                          sig_all_df: pd.DataFrame,
                          plat_all_df: pd.DataFrame,
                          proj_prior_all_df: pd.DataFrame,
                          cos_prior_all_df: pd.DataFrame,
                          out_dir: Path,
                          artifact_dir: Path | None):
    print("\n[Summary] plotting unified all-traits summary heatmap...")

    def avg_dyn(df):
        if df is None or df.empty:
            return pd.DataFrame()
        return df.groupby("val")[["dyn_score", "dyn_ppl"]].mean().reset_index()

    logit_avg      = avg_dyn(logit_all_df)
    anti_avg       = avg_dyn(anti_all_df)
    sig_avg        = avg_dyn(sig_all_df)
    plat_avg       = avg_dyn(plat_all_df)
    proj_prior_avg = avg_dyn(proj_prior_all_df)
    cos_prior_avg  = avg_dyn(cos_prior_all_df)

    out_dir.mkdir(parents=True, exist_ok=True)

    separators = [
        ("DLS_logit_diff", "navy"),
        ("DLS_anti_align", "darkred"),
        ("Fusion_Sigmoid", "darkorange"),
        ("Fusion_Plateau", "purple"),
        ("DLS_proj_prior", "darkcyan"),
        ("DLS_cos_prior", "coral"),
    ]

    method_dfs = [
        ("DLS_logit_diff", logit_avg),
        ("DLS_anti_align", anti_avg),
        ("Fusion_Sigmoid", sig_avg),
        ("Fusion_Plateau", plat_avg),
        ("DLS_proj_prior", proj_prior_avg),
        ("DLS_cos_prior", cos_prior_avg),
    ]

    has_layer_data = not all_df.empty
    if has_layer_data:
        avg = all_df.groupby(["layer", "val"])[
            ["const_score", "const_ppl"]
        ].mean().reset_index()

        p_score = avg.pivot(index="val", columns="layer", values="const_score")
        p_ppl   = avg.pivot(index="val", columns="layer", values="const_ppl")
    else:
        p_score = make_empty_pivot(VALS)
        p_ppl   = make_empty_pivot(VALS)

    for name, avg_df in method_dfs:
        if avg_df.empty:
            continue
        idx = avg_df.set_index("val")
        p_score[name] = idx["dyn_score"]
        p_ppl[name]   = idx["dyn_ppl"]

    p_score = p_score.reindex(VALS)
    p_ppl   = p_ppl.reindex(VALS)

    fig, axes = plt.subplots(2, 1, figsize=(28, 16))
    configs = [
        (axes[0], p_score, p_ppl, "Unified Summary — Score (All traits avg)", "YlGn",     1,   5, ".2f"),
        (axes[1], p_ppl,   p_ppl, "Unified Summary — PPL   (All traits avg)", "RdYlGn_r", 1, 100, ".1f"),
    ]

    for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs:
        sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="gray",
                    ax=ax_obj, annot_kws={"size": 8})

        cols = list(p_data.columns)
        for col_name, color in separators:
            if col_name in cols:
                ax_obj.axvline(x=cols.index(col_name), color=color, linewidth=3.0)

        highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
        ax_obj.set_title(
            f"{title}"
            f" (Black Border: PPL <= 25.0 | Navy/Red: DLS | Orange/Purple: Raw-Norm Fusion | Teal: Proj-Prior)",
            fontsize=12, fontweight="bold")
        ax_obj.set_xlabel("Layer (0 to 31) / Evaluation Variants (rightmost)", fontsize=10)
        ax_obj.set_ylabel("Val (Steering Intensity)", fontsize=10)

    plt.suptitle("Unified 32-Layer Steering & DLS/Fusion Summary (All Traits Average, with Proj-Prior)",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = out_dir / "summary_proj_prior_compare.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary heatmap to: {out_path}")

    # Copy to artifact folder if available
    if artifact_dir and artifact_dir.exists():
        dest = artifact_dir / "summary_proj_prior_compare.png"
        shutil.copy(out_path, dest)
        print(f"  Copied to artifact: {dest}")

    # Plot summary single-layer only heatmap with reversed y-axis
    if has_layer_data:
        print("  Plotting summary single-layer only heatmap (reversed y-axis)...")
        p_score_rev = p_score.reindex(reversed(VALS))
        p_ppl_rev   = p_ppl.reindex(reversed(VALS))

        # Filter columns to keep only the integer layer columns (0 to 31)
        layer_cols = [c for c in p_score_rev.columns if isinstance(c, int) or str(c).isdigit()]
        p_score_only = p_score_rev[layer_cols]
        p_ppl_only = p_ppl_rev[layer_cols]

        fig_s, axes_s = plt.subplots(2, 1, figsize=(22, 14))
        configs_s = [
            (axes_s[0], p_score_only, p_ppl_only, "Unified Summary (Only) — Score (All traits avg)", "YlGn",     1,   5, ".2f"),
            (axes_s[1], p_ppl_only,   p_ppl_only, "Unified Summary (Only) — PPL   (All traits avg)", "RdYlGn_r", 1, 100, ".1f"),
        ]

        for ax_obj, p_data, p_ppl_ref, title, cmap, vmin, vmax, fmt in configs_s:
            sns.heatmap(p_data, annot=True, fmt=fmt, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        linewidths=0.4, linecolor="gray",
                        ax=ax_obj, annot_kws={"size": 8})

            highlight_safe_cells(ax_obj, p_ppl_ref, threshold=25.0)
            ax_obj.set_title(
                f"{title} (Black Border: PPL <= 25.0)",
                fontsize=12, fontweight="bold")
            ax_obj.set_xlabel("Layer (0 to 31)", fontsize=10)
            ax_obj.set_ylabel("Val (Steering Intensity)", fontsize=10)

        plt.suptitle("Unified 32-Layer Steering Summary (Only, All Traits Average - Reversed Steering Intensity Axis)",
                     fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()

        out_path_s = out_dir / "summary_single_layer_only.png"
        plt.savefig(out_path_s, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    Saved summary single-layer only heatmap to: {out_path_s}")

        if artifact_dir and artifact_dir.exists():
            dest_s = artifact_dir / "summary_single_layer_only.png"
            shutil.copy(out_path_s, dest_s)
            print(f"    Copied to artifact: {dest_s}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",      default="exp_steering_layer_analysis/results")
    ap.add_argument("--all_layers_dir", default="exp_steering_dyn_layer_all_layers_midpoint/results")
    ap.add_argument("--fusion_dir",     default="exp_steering_dyn_ic_fusion_midpoint/results")
    ap.add_argument("--proj_prior_dir", default="exp_steering_dyn_layer_proj_prior/results")
    ap.add_argument("--out_dir",        default="exp_steering_dyn_layer_proj_prior/figures")
    ap.add_argument("--artifact_dir",   default="/home/s2550009/.gemini/antigravity-ide/brain/42af965e-7b98-48aa-bc1b-ea07d6f49983/images")
    args = ap.parse_args()

    input_dir      = Path(args.input_dir)
    all_layers_dir = Path(args.all_layers_dir)
    fusion_dir     = Path(args.fusion_dir)
    proj_prior_dir = Path(args.proj_prior_dir)
    out_dir        = Path(args.out_dir)
    artifact_dir   = Path(args.artifact_dir) if args.artifact_dir else None

    all_dfs            = []
    all_logit_dfs      = []
    all_anti_dfs       = []
    all_sig_dfs        = []
    all_plat_dfs       = []
    all_proj_prior_dfs = []
    all_cos_prior_dfs  = []

    for axis in TRAITS:
        df = load_summary(input_dir, axis)
        if not df.empty:
            all_dfs.append(df)

        logit_df = load_dyn_summary(all_layers_dir, axis, "logit_diff")
        if not logit_df.empty:
            all_logit_dfs.append(logit_df)

        anti_df = load_dyn_summary(all_layers_dir, axis, "anti_alignment")
        if not anti_df.empty:
            all_anti_dfs.append(anti_df)

        sig_df = load_fusion_summary(fusion_dir, axis, "sigmoid")
        if not sig_df.empty:
            all_sig_dfs.append(sig_df)

        plat_df = load_fusion_summary(fusion_dir, axis, "soft_plateau")
        if not plat_df.empty:
            all_plat_dfs.append(plat_df)

        proj_prior_df = load_proj_prior_summary(proj_prior_dir, axis)
        if not proj_prior_df.empty:
            all_proj_prior_dfs.append(proj_prior_df)

        cos_prior_df = load_cos_prior_summary(proj_prior_dir, axis)
        if not cos_prior_df.empty:
            all_cos_prior_dfs.append(cos_prior_df)

        plot_axis(df, axis, out_dir / axis, all_layers_dir, fusion_dir, proj_prior_dir, artifact_dir)

    all_df         = pd.concat(all_dfs,            ignore_index=True) if all_dfs            else pd.DataFrame()
    logit_all_df   = pd.concat(all_logit_dfs,      ignore_index=True) if all_logit_dfs      else pd.DataFrame()
    anti_all_df    = pd.concat(all_anti_dfs,       ignore_index=True) if all_anti_dfs       else pd.DataFrame()
    sig_all_df     = pd.concat(all_sig_dfs,        ignore_index=True) if all_sig_dfs        else pd.DataFrame()
    plat_all_df    = pd.concat(all_plat_dfs,       ignore_index=True) if all_plat_dfs       else pd.DataFrame()
    proj_prior_all = pd.concat(all_proj_prior_dfs, ignore_index=True) if all_proj_prior_dfs else pd.DataFrame()
    cos_prior_all  = pd.concat(all_cos_prior_dfs,  ignore_index=True) if all_cos_prior_dfs  else pd.DataFrame()

    make_summary_heatmaps(all_df, logit_all_df, anti_all_df, sig_all_df, plat_all_df, proj_prior_all, cos_prior_all, out_dir, artifact_dir)
    print("\nAll layers projection-prior unified heatmap generation finished.")

if __name__ == "__main__":
    main()
