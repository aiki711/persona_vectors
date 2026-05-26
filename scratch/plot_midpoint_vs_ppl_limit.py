import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
v_bank = 'exp_steering_layer_sweep/vectors/mean_diff_vectors.npz'
data = np.load(v_bank)

traits = ['extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness']
layers = list(range(32))
vals = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

# 1. Compute average midpoint norms for all 32 layers
mp_norms = []
for L in layers:
    layer_mp_norms = []
    for t in traits:
        key = f'{L}|{t}|midpoint'
        if key in data:
            layer_mp_norms.append(np.linalg.norm(data[key]))
    mp_norms.append(np.mean(layer_mp_norms) if layer_mp_norms else 0.0)

# 2. Find max alpha where PPL <= 25.0
single_layer_dir = Path('exp_steering_layer_analysis/results')
max_safe_alphas = []

for L in layers:
    max_val = 0.0
    for val in vals:
        ppls = []
        for t in traits:
            csv_path = single_layer_dir / t / f"scores_layer_{L}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = single_layer_dir / t / f"scores_layer_{L}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    ppls.extend(df['const_ppl'].tolist())
                except Exception as e:
                    pass
        if ppls:
            avg_ppl = np.mean(ppls)
            if avg_ppl <= 25.0:
                max_val = val
    max_safe_alphas.append(max_val)

# 3. Create the dual-axis plot
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(14, 7))

color = '#1f77b4'
ax1.set_xlabel('Layer Number', fontsize=12, labelpad=10)
ax1.set_ylabel('Midpoint L2 Norm (Left Axis)', color=color, fontsize=12, labelpad=10)
# Plot line for midpoint norms
line1 = ax1.plot(layers, mp_norms, marker='o', linewidth=2.5, color=color, label='Midpoint L2 Norm')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(layers)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = '#e31a1c'
ax2.set_ylabel('Max Safe Alpha (PPL <= 25) (Right Axis)', color=color, fontsize=12, labelpad=10)
# Plot bars for max safe alpha
bars = ax2.bar(layers, max_safe_alphas, alpha=0.35, color=color, width=0.5, label='Max Safe Alpha (PPL <= 25)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 45) # Max alpha is 40
ax2.grid(False) # Prevent overlapping grid lines

# Annotate bars with their values (rotated 90 degrees for readability with 32 bars)
for i, val in enumerate(max_safe_alphas):
    ax2.text(layers[i], val + 0.8, f'{val}', ha='center', va='bottom', color=color, fontsize=8.5, fontweight='bold', rotation=90)

# Title
plt.title('Relationship between Layer Midpoint Norm and Maximum Safe Alpha (PPL <= 25)\n(Mistral-7B Full 32-Layer Single-Layer Constant Alpha Steering Sweep)', fontsize=14, fontweight='bold', pad=15)

# Add legends
lines = line1 + [bars]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()

# Save to artifacts
out_dir = Path('/home/s2550009/.gemini/antigravity-ide/brain/42af965e-7b98-48aa-bc1b-ea07d6f49983/images')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'midpoint_vs_max_safe_alpha.png'
plt.savefig(out_path, dpi=200)
plt.close()

# Also calculate correlation coefficient
corr = np.corrcoef(mp_norms, max_safe_alphas)[0, 1]
print(f"Midpoint norms: {[round(x, 2) for x in mp_norms]}")
print(f"Max safe alphas: {max_safe_alphas}")
print(f"Correlation coefficient (Pearson R): {corr:.4f}")
print(f"Saved plot to: {out_path}")
