import numpy as np

v_bank = 'exp_steering_layer_sweep/vectors/mean_diff_vectors.npz'
data = np.load(v_bank)

traits = ['extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness']
layers = range(32)

output = []
output.append('| Layer | Midpoint L2 Norm (Avg) | w L2 Norm (Extraversion) | w L2 Norm (Neuroticism) | w L2 Norm (Agreeableness) |')
output.append('| :---: | :---: | :---: | :---: | :---: |')

for L in layers:
    # Midpoint norm
    mp_norms = []
    for t in traits:
        key = f'{L}|{t}|midpoint'
        if key in data:
            mp_norms.append(np.linalg.norm(data[key]))
    avg_mp_norm = np.mean(mp_norms) if mp_norms else float('nan')
    
    # Steering vector norms
    w_ext = np.linalg.norm(data[f'{L}|extraversion|w']) if f'{L}|extraversion|w' in data else float('nan')
    w_neu = np.linalg.norm(data[f'{L}|neuroticism|w']) if f'{L}|neuroticism|w' in data else float('nan')
    w_agr = np.linalg.norm(data[f'{L}|agreeableness|w']) if f'{L}|agreeableness|w' in data else float('nan')
    
    output.append(f'| **Layer {L}** | {avg_mp_norm:.3f} | {w_ext:.3f} | {w_neu:.3f} | {w_agr:.3f} |')

with open('scratch/norms_table.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output) + '\n')

print("DONE")
