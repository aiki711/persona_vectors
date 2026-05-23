import glob
output = []
for trait in ['extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness']:
    files = glob.glob(f'exp_steering_dyn_ic_fusion_midpoint/results/{trait}/scores_*.csv')
    output.append(f'{trait}: {len(files)} / 42 csv files generated')

with open('scratch/progress.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output) + '\n')
print("DONE")
