import json
from pathlib import Path

def print_first_entry(path):
    print(f"=== {path} ===")
    if not Path(path).exists():
        print("File does not exist.")
        return
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            print("Empty file.")
            return
        data = json.loads(line)
        print("Prompt:", repr(data.get("prompt"))[:100])
        print("Selected layer:", data.get("dyn_layer"))
        print("Base PPL:", data.get("base_ppl"))
        print("Dyn PPL:", data.get("dyn_ppl"))
        print("Base Text:", repr(data.get("base_text"))[:100])
        print("Dyn Text:", repr(data.get("dyn_text"))[:100])
        raw_scores = data.get("raw_scores")
        if raw_scores:
            # print first 5 scores
            first_5 = {k: raw_scores[k] for k in sorted(raw_scores.keys(), key=int)[:5]}
            print("Raw scores (first 5 layers):", first_5)

print_first_entry("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/logit_diff_Val1.0.jsonl")
print_first_entry("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/cos_only_Val1.0.jsonl")
print_first_entry("exp_steering_dyn_layer_proj_prior/results/extraversion/cos_only_Val1.0.jsonl")
print_first_entry("exp_steering_dyn_layer_proj_prior/results/extraversion/rank_only_Val1.0.jsonl")
