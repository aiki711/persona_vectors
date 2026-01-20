#!/usr/bin/env python3
# scripts/07_alpha_visualize.py
# Compatible with 06_alpha_eval_v12.py output format

from __future__ import annotations

import argparse, glob, json, os, re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 定数設定
# ---------------------------------------------------------
TRAIT_ORDER = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Base=青, Instruct=オレンジ
COLORS = {"base": "#1f77b4", "instruct": "#ff7f0e"}

FILENAME_RE = re.compile(
    r"alpha_(range|selected)_(base|instruct)_(\w+)\.jsonl$"
)

# ---------------------------------------------------------
# ファイル読み込み・解析
# ---------------------------------------------------------
def parse_filename_meta(path: str) -> Optional[Tuple[str, str, str]]:
    base = os.path.basename(path)
    m = FILENAME_RE.match(base)
    if not m:
        return None
    # kind, split, trait
    return m.group(1), m.group(2), m.group(3).lower()

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return rows

def to_float(x: Any) -> Optional[float]:
    if x is None: return None
    try: return float(x)
    except: return None

# ---------------------------------------------------------
# 集計ロジック (v12対応)
# ---------------------------------------------------------
def summarize_selected(tag: str, split: str, trait: str, fpath: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # v12では "best_alpha_total" ではなく "alpha_total" (選ばれた行そのもの) が入っている
    # 互換性のため両方チェックする
    alphas = []
    dss = []
    
    for r in rows:
        # v12 format check
        a = r.get("best_alpha_total") if "best_alpha_total" in r else r.get("alpha_total")
        d = r.get("best_ds_avg") if "best_ds_avg" in r else r.get("ds_avg")
        
        val_a = to_float(a)
        val_d = to_float(d)
        
        if val_a is not None: alphas.append(val_a)
        if val_d is not None: dss.append(val_d)

    return {
        "tag": tag, "kind": "selected", "split": split, "trait": trait,
        "n_samples": len(rows),
        "selected_alpha_mean": (sum(alphas)/len(alphas)) if alphas else None,
        "selected_alpha_median": (pd.Series(alphas).median() if alphas else None),
        "best_ds_mean": (sum(dss)/len(dss)) if dss else None,
        "best_ds_median": (pd.Series(dss).median() if dss else None),
    }

def summarize_range(tag: str, split: str, trait: str, fpath: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows: return {}
    
    # v12 check: 1行目にメタデータと stats がネストされているか確認
    first_row = rows[0]
    
    # --- v12 Format Handler ---
    if "per_alpha_stats" in first_row and ("alpha_recommended" in first_row or "alpha_rec" in first_row):
        # v12 structure
        rec_val = to_float(first_row.get("alpha_recommended"))
        range_lo = to_float(first_row.get("alpha_lo"))
        range_hi = to_float(first_row.get("alpha_hi"))
        
        # 統計情報から推奨値のmedianなどを取得
        stats_list = first_row.get("per_alpha_stats", [])
        rec_stats = next((s for s in stats_list if to_float(s.get("alpha_total")) == rec_val), None)
        
        return {
            "tag": tag, "kind": "range", "split": split, "trait": trait,
            "n_alphas": len(stats_list),
            "range_lo": range_lo,
            "range_hi": range_hi,
            "recommended": rec_val,
            "rec_median": to_float(rec_stats.get("median")) if rec_stats else None,
            "rec_mean": to_float(rec_stats.get("mean")) if rec_stats else None,
            "rec_p_pos": to_float(rec_stats.get("p_pos")) if rec_stats else None,
        }

    # --- Legacy (v11) Format Handler ---
    # フラットなリスト構造の場合
    rec_row = next((r for r in rows if r.get("is_recommended")), None)
    recommended = to_float(first_row.get("recommended"))
    # Fallback logic
    if rec_row is None and recommended is not None:
        rec_row = next((r for r in rows if to_float(r.get("alpha_total")) == recommended), None)

    return {
        "tag": tag, "kind": "range", "split": split, "trait": trait,
        "n_alphas": len(rows),
        "range_lo": to_float(first_row.get("range_lo")),
        "range_hi": to_float(first_row.get("range_hi")),
        "recommended": recommended,
        "rec_median": to_float(rec_row.get("median")) if rec_row else None,
        "rec_mean": to_float(rec_row.get("mean")) if rec_row else None,
        "rec_p_pos": to_float(rec_row.get("p_pos")) if rec_row else None,
    }

# ---------------------------------------------------------
# 可視化ロジック
# ---------------------------------------------------------
def plot_range_comparison(df: pd.DataFrame, tag: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = range(len(TRAIT_ORDER))
    offsets = {"base": -0.15, "instruct": 0.15}
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    has_data = False
    
    for i, trait in enumerate(TRAIT_ORDER):
        df_trait = df[df["trait"] == trait]
        
        for split in ["base", "instruct"]:
            row = df_trait[df_trait["split"] == split]
            if row.empty:
                continue
            
            has_data = True
            lo = row.iloc[0]["range_lo"]
            hi = row.iloc[0]["range_hi"]
            rec = row.iloc[0]["recommended"]
            
            if pd.isna(lo) or pd.isna(hi):
                continue

            y = i + offsets[split]
            color = COLORS.get(split, "gray")
            
            # Range Line
            ax.hlines(y, lo, hi, colors=color, linewidth=3, alpha=0.6)
            ax.vlines(lo, y - 0.1, y + 0.1, colors=color, linewidth=2)
            ax.vlines(hi, y - 0.1, y + 0.1, colors=color, linewidth=2)
            
            # Recommended Dot
            label_text = split.capitalize() if i == 0 else ""
            ax.plot(rec, y, 'o', color=color, markersize=9, label=label_text, zorder=5)

    if not has_data:
        plt.close(fig)
        return

    ax.set_yticks(y_positions)
    ax.set_yticklabels([t.capitalize() for t in TRAIT_ORDER], fontsize=12)
    ax.invert_yaxis()
    
    ax.set_xlabel("Alpha Value", fontsize=12)
    ax.set_title(f"Alpha Range & Recommendation\nModel: {tag}", fontsize=14)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Saved Range Plot: {os.path.basename(out_path)}")

def plot_grouped_bar(df_sub: pd.DataFrame, metric: str, title: str, out_path: str):
    plt.figure(figsize=(10, 6))
    
    df_clean = df_sub.dropna(subset=[metric])
    if df_clean.empty:
        plt.close()
        return

    sns.barplot(
        data=df_clean,
        x="trait",
        y=metric,
        hue="split",
        order=TRAIT_ORDER,
        palette=COLORS,
        edgecolor="black",
        alpha=0.8
    )
    
    plt.title(title, fontsize=14)
    plt.ylabel(metric)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title="Model Split")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved Bar Chart:  {os.path.basename(out_path)}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", nargs="+", required=True, help="Input file patterns")
    ap.add_argument("--out_csv", required=True, help="Path to save summary CSV")
    ap.add_argument("--out_dir", required=True, help="Directory to save plots")
    ap.add_argument("--compare_splits", action="store_true", help="(Always enabled implicitly)")
    args = ap.parse_args()

    files = []
    for g in args.globs:
        files.extend(glob.glob(g, recursive=True))
    files = sorted(set(files))
    
    if not files:
        raise SystemExit(f"[ERROR] No files matched.")
    
    print(f"[INFO] Found {len(files)} files. Processing...")

    summary_rows = []
    raw_rows_selected = [] # Optional: to debug raw data extraction
    
    for fpath in files:
        meta = parse_filename_meta(fpath)
        if not meta: continue
        kind, split, trait = meta
        
        if trait not in TRAIT_ORDER: continue
        
        rows = read_jsonl(fpath)
        if not rows: continue
        
        # tag取得 (v12の場合はrows[0]にtagがある)
        tag = rows[0].get("tag", "unknown")
        
        if kind == "selected":
            summary_rows.append(summarize_selected(tag, split, trait, fpath, rows))
        elif kind == "range":
            summary_rows.append(summarize_range(tag, split, trait, fpath, rows))

    df = pd.DataFrame(summary_rows)
    if df.empty:
        raise SystemExit("[ERROR] No valid data extracted.")

    df["trait"] = pd.Categorical(df["trait"], categories=TRAIT_ORDER, ordered=True)
    df = df.sort_values(["tag", "kind", "split", "trait"])
    
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Summary CSV saved: {args.out_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    unique_tags = df["tag"].unique()
    
    for tag in unique_tags:
        print(f"--- Visualizing Model: {tag} ---")
        df_tag = df[df["tag"] == tag]
        
        # (A) Range Data -> Range Plot
        df_rng = df_tag[df_tag["kind"] == "range"]
        if not df_rng.empty:
            out_name = f"range_plot_{tag}.png"
            plot_range_comparison(df_rng, tag, os.path.join(args.out_dir, out_name))
        
        # (B) Selected Data -> Bar Chart
        df_sel = df_tag[df_tag["kind"] == "selected"]
        if not df_sel.empty:
            metric = "best_ds_mean"
            plot_grouped_bar(
                df_sel, metric, 
                f"Steering Effectiveness (Delta Score)\nModel: {tag}",
                os.path.join(args.out_dir, f"bar_{tag}_{metric}.png")
            )
            metric_a = "selected_alpha_median"
            plot_grouped_bar(
                df_sel, metric_a, 
                f"Required Intervention Strength (Alpha)\nModel: {tag}",
                os.path.join(args.out_dir, f"bar_{tag}_{metric_a}.png")
            )

    print(f"[OK] All plots saved to: {args.out_dir}")

if __name__ == "__main__":
    main()