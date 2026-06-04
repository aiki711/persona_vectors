import json
from pathlib import Path

rank_file = Path("exp_steering_dyn_layer_proj_prior/results/extraversion/rank_only_Val5.0.jsonl")
cos_file = Path("exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/cos_only_Val5.0.jsonl")

if not rank_file.exists():
    print(f"Error: Rank file does not exist: {rank_file}")
    exit(1)
if not cos_file.exists():
    print(f"Error: Cos-Only file does not exist: {cos_file}")
    exit(1)

rank_data = []
with open(rank_file, "r", encoding="utf-8") as f:
    for line in f:
        rank_data.append(json.loads(line))

cos_data = []
with open(cos_file, "r", encoding="utf-8") as f:
    for line in f:
        cos_data.append(json.loads(line))

print(f"Number of prompts: Rank={len(rank_data)}, Cos-Only={len(cos_data)}")

print("\nPrompt-by-Prompt Layer Selection Comparison (Alpha=5.0, Extraversion):")
print("-" * 80)
print(f"{'Idx':3} | {'Prompt Snippet':40} | {'Rank-based Layer':16} | {'Cos-Only Layer':16}")
print("-" * 80)

for i in range(min(len(rank_data), len(cos_data))):
    r_row = rank_data[i]
    c_row = cos_data[i]
    
    # Verify prompts match
    r_prompt = r_row["prompt"]
    c_prompt = c_row["prompt"]
    prompts_match = r_prompt.strip() == c_prompt.strip()
    
    snippet = c_prompt[:37] + "..."
    r_layer = r_row["dyn_layer"]
    c_layer = c_row["dyn_layer"]
    
    match_str = " (Mismatch!)" if not prompts_match else ""
    print(f"{i:3d} | {snippet:40} | L{r_layer:<15d} | L{c_layer:<15d}{match_str}")
