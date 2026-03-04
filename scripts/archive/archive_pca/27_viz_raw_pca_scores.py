import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast

def plot_alpha_vs_score(df, trait_name, output_dir):
    """
    Plots alpha_total vs raw_score for a given trait.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate mean and std for each alpha_total
    summary = df.groupby('alpha_total')['raw_score_' + trait_name].agg(['mean', 'std']).reset_index()
    
    sns.lineplot(data=df, x='alpha_total', y='raw_score_' + trait_name, marker='o', label='Individual Prompts', alpha=0.3)
    plt.errorbar(summary['alpha_total'], summary['mean'], yerr=summary['std'], fmt='-s', color='red', label='Mean ± Std', capsize=5)
    
    plt.title(f'Alpha Total vs Raw Score: {trait_name.capitalize()}')
    plt.xlabel('Alpha Total (Steering Intensity)')
    plt.ylabel('Raw Score (0 to 10)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(output_dir / f'{trait_name}_alpha_vs_score.png')
    plt.close()

def plot_layer_contribution(df, trait_name, output_dir):
    """
    Plots the contribution of each layer (s_by_layer) at different alpha levels.
    """
    # Use only entries where alpha_total != 0 for better visualization of steering effect
    steered_df = df[df['alpha_total'] != 0].copy()
    if steered_df.empty:
        return

    # Parse s_by_layer string to dict
    def parse_s_by_layer(val):
        if isinstance(val, str):
            try:
                # Replace single quotes with double quotes for JSON parsing if needed, 
                # but ast.literal_eval is safer for python dict strings
                return ast.literal_eval(val)
            except:
                return {}
        return val

    steered_df['s_by_layer_parsed'] = steered_df['s_by_layer'].apply(parse_s_by_layer)
    
    # Expand the dict into rows
    rows = []
    for _, row in steered_df.iterrows():
        alpha = row['alpha_total']
        for layer, score in row['s_by_layer_parsed'].items():
            rows.append({'alpha': alpha, 'layer': int(layer), 'score': score})
    
    plot_df = pd.DataFrame(rows)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_df, x='layer', y='score', hue='alpha', marker='o', palette='viridis')
    
    plt.title(f'Layer Contribution (s_by_layer): {trait_name.capitalize()}')
    plt.xlabel('Layer')
    plt.ylabel('Steering Score Contribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Alpha Total', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_dir / f'{trait_name}_layer_contribution.png')
    plt.close()

def main():
    base_path = Path("/home/admin/work/s2550009/persona_vectors/exp_raw_pca/mistral_7b/scores")
    plot_dir = base_path / "plot"
    plot_dir.mkdir(exist_ok=True)
    
    files = list(base_path.glob("scores_*.csv"))
    
    for file_path in files:
        trait = file_path.stem.replace("scores_", "")
        print(f"Processing trait: {trait}")
        
        df = pd.read_csv(file_path)
        
        # 1. Alpha vs Score
        if f'raw_score_{trait}' in df.columns:
            plot_alpha_vs_score(df, trait, plot_dir)
        
        # 2. Layer Contribution
        if 's_by_layer' in df.columns:
            plot_layer_contribution(df, trait, plot_dir)

if __name__ == "__main__":
    main()
