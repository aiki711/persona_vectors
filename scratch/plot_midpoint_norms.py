import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
v_bank = 'exp_steering_layer_sweep/vectors/mean_diff_vectors.npz'
data = np.load(v_bank)

traits = ['extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness']
layers = range(32)

# Compute average midpoint norms
mp_norms = []
for L in layers:
    layer_mp_norms = []
    for t in traits:
        key = f'{L}|{t}|midpoint'
        if key in data:
            layer_mp_norms.append(np.linalg.norm(data[key]))
    mp_norms.append(np.mean(layer_mp_norms) if layer_mp_norms else 0.0)

# Set style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot line and points
plt.plot(layers, mp_norms, marker='o', linewidth=2.5, color='#1f77b4', label='Midpoint L2 Norm')
plt.fill_between(layers, mp_norms, color='#1f77b4', alpha=0.15)

# Customize title and labels
plt.title('Midpoint L2 Norm Growth Across Layers (Mistral-7B)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Layer Number', fontsize=11, labelpad=10)
plt.ylabel('Average L2 Norm', fontsize=11, labelpad=10)
plt.xlim(-0.5, 31.5)
plt.xticks(range(0, 32, 2))

# Highlight Layer 1 and Layer 31 values
plt.annotate(f'L1: {mp_norms[1]:.3f}', xy=(1, mp_norms[1]), xytext=(3, mp_norms[1]+1),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
plt.annotate(f'L31: {mp_norms[31]:.3f}', xy=(31, mp_norms[31]), xytext=(26, mp_norms[31]-2),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

# Clean up layout
plt.tight_layout()

# Ensure target directories exist and save
out_dir = Path('/home/s2550009/.gemini/antigravity-ide/brain/42af965e-7b98-48aa-bc1b-ea07d6f49983/images')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'midpoint_norms.png'
plt.savefig(out_path, dpi=200)
plt.close()

print(f"DONE: Saved plot to {out_path}")
