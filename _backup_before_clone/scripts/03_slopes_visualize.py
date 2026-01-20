#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

TRAIT_ORDER = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def _safe_tag(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def _derive_tag(csv_path: str, input_root: str, tag_level: int) -> str:
    """Derive tag from file path (relative to input_root)."""
    try:
        rel = Path(csv_path).resolve().relative_to(Path(input_root).resolve())
        parts = rel.parts
        if not parts:
            return "unknown"
        # tag_level=0 => first component under input_root (recommended: exp/<TAG>/...)
        if tag_level >= 0 and tag_level < len(parts):
            return parts[tag_level]
        # allow negative indexing
        if tag_level < 0 and abs(tag_level) <= len(parts):
            return parts[tag_level]
        return "unknown"
    except Exception:
        return "unknown"


def load_data(input_dir: str, recursive: bool, file_glob: str, tag_level: int) -> pd.DataFrame:
    """Load one CSV or many CSVs from a directory (optionally recursive). Adds `tag` and `_src`."""
    # Single CSV
    if os.path.isfile(input_dir) and input_dir.lower().endswith(".csv"):
        df = pd.read_csv(input_dir)
        df["_src"] = input_dir
        df["tag"] = "single"
        return df

    base = Path(input_dir)
    if not base.exists():
        print(f"[Error] input_dir not found: {input_dir}")
        return pd.DataFrame()

    if recursive:
        pattern = str(base / file_glob)
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = str(base / file_glob.replace("**/", ""))
        files = glob.glob(pattern)

    files = [p for p in files if os.path.isfile(p) and p.lower().endswith(".csv")]
    if not files:
        print(f"[Warning] No CSV files found. pattern={pattern}")
        return pd.DataFrame()

    df_list = []
    print(f"[INFO] Loading {len(files)} files from {input_dir} (recursive={recursive})")
    for f in files:
        try:
            temp = pd.read_csv(f)
            temp["_src"] = f
            temp["tag"] = _derive_tag(f, input_dir, tag_level)
            df_list.append(temp)
        except Exception as e:
            print(f"  [Error] Failed to load {f}: {e}")

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


def plot_layer_sensitivity(df: pd.DataFrame, out_path: str, value_col: str):
    """Layer-wise sensitivity line plots (facet by trait, hue=model)."""
    print(f"  -> Generating: {os.path.basename(out_path)}")
    sns.set_theme(style="whitegrid")

    available_traits = [t for t in TRAIT_ORDER if t in df["trait"].unique()]
    g = sns.FacetGrid(
        df,
        col="trait",
        hue="model",
        col_wrap=3,
        height=4,
        sharey=False,
        col_order=available_traits,
    )
    g.map(sns.lineplot, "layer", value_col, marker="o")
    g.add_legend()
    g.set_axis_labels("Layer", f"Sensitivity ({value_col})")
    g.fig.suptitle("Layer-wise Steering Sensitivity (Base vs Instruct)", y=1.02)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_resistance_heatmap(df: pd.DataFrame, out_path: str, value_col: str):
    """Resistance heatmap (Base - Instruct). Single-tag context recommended."""
    print(f"  -> Generating: {os.path.basename(out_path)}")

    df_base = df[df["model"].str.contains("base", case=False, na=False)]
    df_instr = df[df["model"].str.contains("instruct", case=False, na=False)]

    if df_base.empty or df_instr.empty:
        print("  [Skip] Heatmap requires both 'base' and 'instruct' data.")
        return

    merged = pd.merge(df_base, df_instr, on=["trait", "layer"], suffixes=("_base", "_instr"))
    base_col = f"{value_col}_base"
    instr_col = f"{value_col}_instr"
    if base_col not in merged.columns or instr_col not in merged.columns:
        print(f"  [Skip] Heatmap missing columns after merge: {base_col}, {instr_col}")
        return

    merged["resistance"] = merged[base_col] - merged[instr_col]
    heatmap_data = merged.pivot(index="trait", columns="layer", values="resistance")

    valid_order = [t for t in TRAIT_ORDER if t in heatmap_data.index]
    heatmap_data = heatmap_data.reindex(valid_order)

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="Reds", center=0, annot=False)
    plt.title(
        "Resistance Heatmap (Positive = Instruct Resists Steering)\n"
        f"value={value_col}  (Red = Instruct suppresses vs Base)"
    )
    plt.xlabel("Layer")
    plt.ylabel("Trait")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_trait_avg_sensitivity_bar(df: pd.DataFrame, out_path: str, value_col: str, hue_col: str):
    """Average sensitivity per trait bar plot."""
    print(f"  -> Generating: {os.path.basename(out_path)}")

    avg_df = df.groupby([hue_col, "trait"], as_index=False)[value_col].mean()
    available_traits = [t for t in TRAIT_ORDER if t in avg_df["trait"].unique()]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_df, x="trait", y=value_col, hue=hue_col, order=available_traits)
    plt.title(f"Average Sensitivity per Trait\nvalue={value_col}  hue={hue_col}")
    plt.ylabel(f"Average {value_col}")
    plt.xlabel("Trait")
    plt.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_trait_avg_sensitivity_radar_lines(avg_df: pd.DataFrame, out_path: str, value_col: str, line_col: str):
    """Radar chart: one line per line_col."""
    print(f"  -> Generating: {os.path.basename(out_path)}")

    traits = [t for t in TRAIT_ORDER if t in avg_df["trait"].unique()]
    if not traits:
        print("  [Skip] No traits found for radar.")
        return

    lines = list(avg_df[line_col].unique())
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)

    for l in lines:
        sub = avg_df[avg_df[line_col] == l].set_index("trait").reindex(traits)
        vals = sub[value_col].to_numpy(dtype=float)
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(angles, vals, linewidth=2, label=str(l))
        ax.fill(angles, vals, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits)
    ax.set_title(f"Average Sensitivity per Trait (Radar)\nvalue={value_col}  line={line_col}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_radar_multi_tag_split_base_instruct(df: pd.DataFrame, out_dir: str, value_col: str, suffix: str):
    """Make two radar charts: base (6 tags) and instruct (6 tags)."""
    # Average across layers first
    avg_df = df.groupby(["tag", "model", "trait"], as_index=False)[value_col].mean()

    for m in ["base", "instruct"]:
        sub = avg_df[avg_df["model"].str.lower() == m].copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={"tag": "line"})
        out_path = os.path.join(out_dir, f"trait_sensitivity_radar_{m}_{_safe_tag(value_col)}{suffix}.png")
        plot_trait_avg_sensitivity_radar_lines(sub, out_path, value_col=value_col, line_col="line")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing slopes_*.csv files (or a single CSV)")
    parser.add_argument("--out_dir", required=True, help="Directory to save plot images")
    parser.add_argument(
        "--value_col",
        default="slope_delta_score_vs_alpha",
        help="Metric column to visualize (e.g., slope_delta_score_vs_alpha01)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Load CSVs recursively under input_dir (for comparing multiple TAGs).",
    )
    parser.add_argument(
        "--file_glob",
        default="**/slopes_*.csv",
        help="Glob pattern to select CSVs under input_dir (default: **/slopes_*.csv)",
    )
    parser.add_argument(
        "--tag_level",
        type=int,
        default=0,
        help="Derive tag from relative path parts (0 means first directory under input_dir, e.g., exp/<TAG>/...).",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.input_dir, recursive=args.recursive, file_glob=args.file_glob, tag_level=args.tag_level)
    if df.empty:
        print("[Error] No data loaded. Check input_dir / file_glob.")
        return

    if args.value_col not in df.columns:
        print(f"[Error] value_col not found: {args.value_col}")
        print(f"        Available columns: {list(df.columns)}")
        return

    # numeric conversion
    if "layer" in df.columns:
        df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df[args.value_col] = pd.to_numeric(df[args.value_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # drop NaNs necessary for plotting
    required = ["trait", "model", args.value_col]
    if "layer" in df.columns:
        required.append("layer")
    df = df.dropna(subset=required)

    if "layer" in df.columns:
        df["layer"] = df["layer"].astype(int)

    value_tag = _safe_tag(args.value_col)
    n_tags = df["tag"].nunique() if "tag" in df.columns else 1
    multi_tag = n_tags > 1

    print(f"--- Visualizing Results to {args.out_dir} (value_col={args.value_col}, tags={n_tags}) ---")

    # pooling/axis_mode split
    if "pooling" in df.columns and "axis_mode" in df.columns:
        for (pooling, axis_mode), sub in df.groupby(["pooling", "axis_mode"]):
            suffix = f"_pool-{pooling}_mode-{axis_mode}"
            print(f"[Subset] pooling={pooling}, axis_mode={axis_mode}, n={len(sub)}")

            if multi_tag:
                # requested: base & instruct each have 6 TAG lines
                plot_radar_multi_tag_split_base_instruct(sub, args.out_dir, value_col=args.value_col, suffix=suffix)

                # optional: bars per base/instruct with hue=tag
                for m in ["base", "instruct"]:
                    sub_m = sub[sub["model"].str.lower() == m]
                    if sub_m.empty:
                        continue
                    out_bar = os.path.join(args.out_dir, f"trait_sensitivity_bar_{m}_{value_tag}{suffix}.png")
                    plot_trait_avg_sensitivity_bar(sub_m, out_bar, value_col=args.value_col, hue_col="tag")

            else:
                # original behavior (single TAG): base vs instruct 2 lines
                out_layer = os.path.join(args.out_dir, f"layer_sensitivity_{value_tag}{suffix}.png")
                out_trait = os.path.join(args.out_dir, f"trait_sensitivity_bar_{value_tag}{suffix}.png")
                out_radar = os.path.join(args.out_dir, f"trait_sensitivity_radar_{value_tag}{suffix}.png")
                out_heat = os.path.join(args.out_dir, f"resistance_heatmap_{value_tag}{suffix}.png")

                plot_layer_sensitivity(sub, out_layer, value_col=args.value_col)
                plot_trait_avg_sensitivity_bar(sub, out_trait, value_col=args.value_col, hue_col="model")

                avg_df = sub.groupby(["model", "trait"], as_index=False)[args.value_col].mean()
                avg_df = avg_df.rename(columns={"model": "line"})
                plot_trait_avg_sensitivity_radar_lines(avg_df, out_radar, value_col=args.value_col, line_col="line")

                plot_resistance_heatmap(sub, out_heat, value_col=args.value_col)

    else:
        # no pooling/axis_mode split
        if multi_tag:
            plot_radar_multi_tag_split_base_instruct(df, args.out_dir, value_col=args.value_col, suffix="")
            for m in ["base", "instruct"]:
                df_m = df[df["model"].str.lower() == m]
                if df_m.empty:
                    continue
                out_bar = os.path.join(args.out_dir, f"trait_sensitivity_bar_{m}_{value_tag}.png")
                plot_trait_avg_sensitivity_bar(df_m, out_bar, value_col=args.value_col, hue_col="tag")
        else:
            plot_layer_sensitivity(df, os.path.join(args.out_dir, f"layer_sensitivity_{value_tag}.png"), value_col=args.value_col)
            plot_trait_avg_sensitivity_bar(df, os.path.join(args.out_dir, f"trait_sensitivity_bar_{value_tag}.png"), value_col=args.value_col, hue_col="model")

            avg_df = df.groupby(["model", "trait"], as_index=False)[args.value_col].mean()
            avg_df = avg_df.rename(columns={"model": "line"})
            plot_trait_avg_sensitivity_radar_lines(avg_df, os.path.join(args.out_dir, f"trait_sensitivity_radar_{value_tag}.png"), value_col=args.value_col, line_col="line")

            plot_resistance_heatmap(df, os.path.join(args.out_dir, f"resistance_heatmap_{value_tag}.png"), value_col=args.value_col)

    print("--- Done ---")


if __name__ == "__main__":
    main()
