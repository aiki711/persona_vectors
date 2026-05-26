import glob

traits = ['extraversion', 'neuroticism', 'openness', 'conscientiousness', 'agreeableness']
out = []

out.append("=== 32-Layer Sweep Files ===")
for t in traits:
    count = len(glob.glob(f'exp_steering_layer_analysis/results/{t}/scores_layer_*.csv'))
    out.append(f"{t}: {count} / 448 csv files")

out.append("\n=== Proj-Prior DLS Files ===")
for t in traits:
    count = len(glob.glob(f'exp_steering_dyn_layer_proj_prior/results/{t}/scores_proj_prior_Val*.csv'))
    # Also check if there are raw jsonl files
    count_jsonl = len(glob.glob(f'exp_steering_dyn_layer_proj_prior/results/{t}/proj_prior_Val*.jsonl'))
    out.append(f"{t}: {count} csv files, {count_jsonl} jsonl files")

output_str = "\n".join(out)
print(output_str)

with open("scratch/count_files_out.txt", "w") as f:
    f.write(output_str + "\n")
