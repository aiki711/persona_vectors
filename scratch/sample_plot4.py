import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 1. Define layers
    layers = ['Layer 1', 'Layer 2', '...', 'Layer N']
    
    # 2. Define scores (condition: Layer 1 is the highest)
    scores = [0.33, 0.75, 0.0, 0.48] 
    
    # 3. Apply the NEW color scheme: Layer 1 is red, others are grey, ellipsis transparent.
    # Color Framework: Red for High Scores (Accel needed, simplified to just Layer 1),
    # Grey for Neutralize Scores (All others).
    colors = ['#d63031', '#95a5a6', '#ffffff', '#95a5a6', '#95a5a6'] # Red, Grey, Transparent, Grey, Grey
    edge_colors = ['none'] * 5 # No soft borders needed now.
    
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5, 5))
    
    x_pos = np.arange(len(layers))
    
    # Plot the bar chart
    bars = ax.bar(x_pos, scores, color=colors, edgecolor=edge_colors, width=0.6, zorder=3)
    
    # Annotate values exactly on top of each valid bar (skipping the ellipsis)
    for i, bar in enumerate(bars):
        score_val = scores[i]
        if layers[i] != '...':
            ax.text(bar.get_x() + bar.get_width()/2.0, score_val + 0.02, 
                    f"{score_val:.2f}", ha='center', va='bottom', fontsize=15, fontweight='bold', color='#2c3e50')
        else:
            # stylized text for the omission
            ax.text(x_pos[i], 0.25, "...", ha='center', va='center', fontsize=20, fontweight='bold', color='#b0b0b0')
            
    # Chart styling
    ax.set_title("Score Distribution", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Layer", fontsize=15, labelpad=10)
    ax.set_ylabel("Score", fontsize=15, labelpad=10)
    
    # Configure X-axis ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, fontsize=13, fontweight='bold', color='#2c3e50')
    
    # Bounds and Gridlines
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    
    # Clean up borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    plt.tight_layout()
    
    # Save chart
    out_dir = Path("scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proj_rank_layer_scores_bar.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    # print(f"Saved modified layer bar chart to: {out_path}")
    
    # Copy to artifact path (must be exact)
    artifact_dir = Path("/home/s2550009/.gemini/antigravity-ide/brain/6611299f-19cb-4461-bbfe-1854feeb8fae")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dest_path = artifact_dir / "proj_rank_layer_scores_bar.png"
    import shutil
    shutil.copy(out_path, dest_path)
    # print(f"Copied to artifact path: {dest_path}")

if __name__ == "__main__":
    main()