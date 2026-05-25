import glob

traits = ['extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness']
out = []
for t in traits:
    count = len(glob.glob(f'exp_steering_layer_analysis/results/{t}/scores_layer_*.csv'))
    out.append(f"{t}: {count}")

with open("scratch/count_files_out.txt", "w") as f:
    f.write("\n".join(out) + "\n")
print("DONE")
