#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def load_jsonl(file_path: Path) -> pd.DataFrame:
    data = []
    if not file_path.exists():
        return pd.DataFrame()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                # 壊れた行はスキップ
                pass
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing probe_base_*.jsonl files",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save plots",
    )
    ap.add_argument(
        "--norm_path",
        type=Path,
        default=None,
        help=(
            "JSONL file from 02_compare_vector_norms.py "
            "(if omitted, tries ../norm_comparison_all_layers.jsonl)"
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ----- Norm / cosine のファイル読込 -----
    norm_path = args.norm_path
    if norm_path is None:
        # デフォルト: results_dir の 1 つ上 (exp/TAG/) を想定
        candidate = args.results_dir.parent / "norm_comparison_all_layers.jsonl"
        if candidate.exists():
            norm_path = candidate

    df_norm_all = pd.DataFrame()
    have_norm = False
    if norm_path is not None and norm_path.exists():
        df_norm_all = load_jsonl(norm_path)
        if not df_norm_all.empty:
            have_norm = True
            print(f"[INFO] Loaded norm/cosine comparison from {norm_path}")
        else:
            print(f"[WARN] Norm file {norm_path} is empty.")
    else:
        print("[INFO] Norm comparison file not found; plotting steering effect only.")

    # ----- Figure 構成 -----
    if have_norm:
        # 3 行 x 5 列
        fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=False)
        axes_top = axes[0]
        axes_mid = axes[1]
        axes_bottom = axes[2]
        fig.suptitle(
            "Steering Effect, Norm Ratio, and Cosine Similarity - All Traits",
            fontsize=18,
            y=1.03,
        )
    else:
        # 従来通り 1 行 x 5 列
        fig, axes_top = plt.subplots(1, 5, figsize=(14, 4), sharey=True)
        axes_mid = axes_bottom = None
        fig.suptitle(
            "Steering Effect Comparison - All Traits",
            fontsize=18,
            y=1.05,
        )

    # ----- Trait ごとにループ -----
    for i, trait in enumerate(TRAITS):
        # -------------------------
        # 上段: α vs ds_avg
        # -------------------------
        ax_top = axes_top[i]

        base_path = args.results_dir / f"probe_base_{trait}.jsonl"
        instr_path = args.results_dir / f"probe_instruct_{trait}.jsonl"

        df_base = load_jsonl(base_path)
        df_instr = load_jsonl(instr_path)

        if df_base.empty or df_instr.empty:
            ax_top.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax_top.set_title(trait.capitalize())
        else:
            df_base["Model"] = "Base"
            df_instr["Model"] = "Instruct"
            df_combined = pd.concat([df_base, df_instr], ignore_index=True)

            sns.regplot(
                data=df_combined[df_combined["Model"] == "Base"],
                x="alpha_total",
                y="ds_avg",
                ax=ax_top,
                label="Base",
                scatter_kws={"alpha": 0.3},
                line_kws={"linestyle": "--"},
            )
            sns.regplot(
                data=df_combined[df_combined["Model"] == "Instruct"],
                x="alpha_total",
                y="ds_avg",
                ax=ax_top,
                label="Instruct",
                scatter_kws={"alpha": 0.3},
                line_kws={"linestyle": ":"},
            )

            ax_top.set_title(trait.capitalize(), fontsize=14, fontweight="bold")
            ax_top.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax_top.set_xlabel("Alpha")

            if i == 0:
                ax_top.set_ylabel("Score Change (ds_avg)", fontsize=12)
                ax_top.legend()
            else:
                ax_top.set_ylabel("")

        # -------------------------
        # 中段: Norm ratio (Instr/Base)
        # -------------------------
        if have_norm and axes_mid is not None:
            ax_mid = axes_mid[i]
            df_norm_trait = df_norm_all[df_norm_all["axis"] == trait]

            if df_norm_trait.empty:
                ax_mid.text(0.5, 0.5, "No Norm Data", ha="center", va="center")
                ax_mid.set_title(f"{trait.capitalize()} (Norm ratio)")
            else:
                df_norm_trait = df_norm_trait.sort_values("layer")

                sns.lineplot(
                    data=df_norm_trait,
                    x="layer",
                    y="norm_ratio_instruct_to_base",
                    marker="o",
                    ax=ax_mid,
                )
                ax_mid.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
                ax_mid.set_title(f"{trait.capitalize()} (Norm ratio)", fontsize=12)
                ax_mid.set_xlabel("Layer")

                if i == 0:
                    ax_mid.set_ylabel("Instr/Base Norm", fontsize=10)
                else:
                    ax_mid.set_ylabel("")

        # -------------------------
        # 下段: Cosine similarity
        # -------------------------
        if have_norm and axes_bottom is not None:
            ax_bot = axes_bottom[i]
            df_norm_trait = df_norm_all[df_norm_all["axis"] == trait]

            if df_norm_trait.empty:
                ax_bot.text(0.5, 0.5, "No Cosine Data", ha="center", va="center")
                ax_bot.set_title(f"{trait.capitalize()} (Cosine)")
            else:
                df_norm_trait = df_norm_trait.sort_values("layer")

                if "cosine_similarity" not in df_norm_trait.columns:
                    ax_bot.text(0.5, 0.5, "cosine_similarity not found", ha="center", va="center")
                    ax_bot.set_title(f"{trait.capitalize()} (Cosine)")
                else:
                    sns.lineplot(
                        data=df_norm_trait,
                        x="layer",
                        y="cosine_similarity",
                        marker="o",
                        ax=ax_bot,
                    )
                    ax_bot.axhline(0.0, color="gray", linestyle="--", linewidth=0.5)
                    ax_bot.set_title(f"{trait.capitalize()} (Cosine)", fontsize=12)
                    ax_bot.set_xlabel("Layer")

                    if i == 0:
                        ax_bot.set_ylabel("cos(Base,Instr)", fontsize=10)
                    else:
                        ax_bot.set_ylabel("")

    plt.tight_layout()
    out_path = args.out_dir / "steering_effect_norm_cos_all_traits.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[SAVE] Saved combined plot (with norm & cosine) to {out_path}")


if __name__ == "__main__":
    main()

#python scripts/03_plot_results.py \
#  --results_dir "exp/mistral_7b/results" \
#  --out_dir "exp/mistral_7b/results/plots"