import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse

def plot_internal_states_updated(csv_path):
    print(f"Processing: {csv_path}")
    if os.path.getsize(csv_path) == 0:
        print(f"Skipping empty file: {csv_path}")
        return

    # データの読み込み
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty or invalid CSV: {csv_path}")
        return

    # Check if necessary columns exist
    required_columns = ['Layer', 'Alpha', 'Trait', 'Sim_Global', 'Sim_Proc', 'Norm_Global']
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {csv_path}: Missing required columns. Found {df.columns.tolist()}")
        return

    if df.empty:
        print(f"Skipping empty dataframe: {csv_path}")
        return


    # プロンプト間で平均をとる
    # Group by Layer, Alpha, Trait and take mean of numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Exclude grouping keys from numeric_cols to avoid ambiguity/duplication
    group_keys = ['Layer', 'Alpha', 'Trait']
    value_cols = [c for c in numeric_cols if c not in group_keys]
    
    df_avg = df.groupby(group_keys)[value_cols].mean().reset_index()
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    trait = df['Trait'].iloc[0]
    # Dataset name inference (e.g. from path)
    dataset_name = os.path.basename(os.path.dirname(csv_path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
    
    fig.suptitle(f"Internal State Analysis: {model_name} / {dataset_name} / {trait}\n(Local & Global Debugging Framework)", fontsize=20)

    # 1. ③ Global Delta (累積の成果) - Similarity
    sns.lineplot(data=df_avg, x='Layer', y='Sim_Global', hue='Alpha', palette='coolwarm', ax=axes[0, 0], marker='o')
    axes[0, 0].set_title("③ Global Delta: 累積の成果 (Similarity)")
    axes[0, 0].set_ylabel("Cosine Similarity to Steering Vector")
    
    # 2. ② Processing Delta (モデルの反応) - Similarity
    sns.lineplot(data=df_avg, x='Layer', y='Sim_Proc', hue='Alpha', palette='coolwarm', ax=axes[0, 1], marker='o')
    axes[0, 1].set_title("② Processing Delta: モデルの反応/再解釈 (Similarity)")
    axes[0, 1].set_ylabel("Cosine Similarity (Processing Change)")
    axes[0, 1].axhline(0, color='black', linestyle='--')
    
    # 3. ④ Marginal Gain (純粋な利益・損失)
    if 'Marginal_Sim_Global' in df_avg.columns:
        sns.lineplot(data=df_avg, x='Layer', y='Marginal_Sim_Global', hue='Alpha', palette='coolwarm', ax=axes[1, 0], marker='o')
        axes[1, 0].set_title("④ Marginal Gain: 純粋な利益・損失")
        axes[1, 0].set_ylabel("Difference of Global Similarity")
        axes[1, 0].axhline(0, color='black', linestyle='--')
    else:
        axes[1, 0].set_title("④ Marginal Gain: (Not found in data)")
    
    # 4. ③ Global Delta (累積の乖離強度) - Norm
    sns.lineplot(data=df_avg, x='Layer', y='Norm_Global', hue='Alpha', palette='coolwarm', ax=axes[1, 1], marker='o')
    axes[1, 1].set_title("③ Global Delta: 累積の乖離強度 (Norm)")
    axes[1, 1].set_ylabel("Norm of Difference Vector")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Output logic
    # advice/plot or writing/plot
    base_dir = os.path.dirname(csv_path)
    plot_dir = os.path.join(base_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = os.path.basename(csv_path).replace('.csv', '.png')
    out_path = os.path.join(plot_dir, filename)
    
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    root_dir = "analysis_results/internal_states"
    print(f"Searching for CSVs in {root_dir}...")
    
    files = glob.glob(f"{root_dir}/**/*.csv", recursive=True)
    print(f"Found {len(files)} CSV files.")
    
    for f in files:
        if "internal_states" in f:
            plot_internal_states_updated(f)

if __name__ == "__main__":
    main()
