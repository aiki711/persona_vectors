import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    base_path = "exp_adaptive_steering/results/full_layer_granular"
    traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
    vals = [0.03, 0.06, 0.09, 0.12, 0.15]
    
    results = []
    
    for trait in traits:
        for val in vals:
            # Match files like scores_extraversion_Val0.03.csv
            pattern = os.path.join(base_path, trait, f"scores_{trait}_Val{val}.csv")
            files = glob.glob(pattern)
            if not files:
                # Try fallback for formatting issues if any (e.g. 0.03 -> 0.030 or similar)
                continue
                
            df = pd.read_csv(files[0])
            avg_base = df['base_score'].mean()
            avg_const = df['const_score'].mean()
            avg_adapt = df['adapt_score'].mean()
            
            results.append({
                'trait': trait,
                'val': val,
                'base': avg_base,
                'constant': avg_const,
                'adaptive': avg_adapt
            })
            
    summary_df = pd.DataFrame(results)
    if summary_df.empty:
        print("No data found to plot.")
        return

    # Visual Style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Plot 1: Individual Traits
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    for i, trait in enumerate(traits):
        ax = axes[i]
        subset = summary_df[summary_df['trait'] == trait]
        
        ax.plot(subset['val'], subset['base'], 'k--', label='Base', marker='o', alpha=0.5)
        ax.plot(subset['val'], subset['constant'], color='#3498db', label='Constant Steering', marker='s', linewidth=2)
        ax.plot(subset['val'], subset['adaptive'], color='#e74c3c', label='Adaptive Steering', marker='^', linewidth=2.5)
        
        ax.set_title(trait.capitalize(), fontsize=14, fontweight='bold')
        ax.set_xlabel('Steering Strength (Val)')
        if i % 3 == 0:
            ax.set_ylabel('Target Trait Score (1-5)')
        ax.set_ylim(1, 5)
        ax.legend()

    # Plot 2: Average across all traits (Last subplot)
    ax_avg = axes[5]
    avg_by_val = summary_df.groupby('val')[['base', 'constant', 'adaptive']].mean().reset_index()
    ax_avg.plot(avg_by_val['val'], avg_by_val['base'], 'k--', label='Base', marker='o', alpha=0.5)
    ax_avg.plot(avg_by_val['val'], avg_by_val['constant'], color='#3498db', label='Constant Avg', marker='s', linewidth=2)
    ax_avg.plot(avg_by_val['val'], avg_by_val['adaptive'], color='#e74c3c', label='Adaptive Avg', marker='^', linewidth=2.5)
    ax_avg.set_title("Average Across All Traits", fontsize=14, fontweight='bold', color='navy')
    ax_avg.set_xlabel('Steering Strength (Val)')
    ax_avg.set_ylim(1, 5)
    ax_avg.legend()

    plt.tight_layout()
    plot_path = os.path.join(base_path, "sweep_results_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

    # Display some stats in text
    print("\n--- Summary Statistics (Average across all traits) ---")
    print(avg_by_val.to_string(index=False))

if __name__ == "__main__":
    main()
