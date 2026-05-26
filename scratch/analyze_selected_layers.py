import json
import collections

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

print("=== Logit-Diff Selected Layers at Val 2.0 ===")
for trait in TRAITS:
    p = f"exp_steering_dyn_layer_all_layers_midpoint/results/{trait}/logit_diff_Val2.0.jsonl"
    try:
        layers = []
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                layers.append(d["dyn_layer"])
        counts = collections.Counter(layers)
        print(f"  {trait:18}: {dict(sorted(counts.items()))}")
    except Exception as e:
        print(f"  {trait:18}: Failed to read ({e})")

print("\n=== Proj-Prior Selected Layers at Val 2.0 ===")
for trait in TRAITS:
    p = f"exp_steering_dyn_layer_proj_prior/results/{trait}/proj_prior_Val2.0.jsonl"
    try:
        layers = []
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                layers.append(d["dyn_layer"])
        counts = collections.Counter(layers)
        print(f"  {trait:18}: {dict(sorted(counts.items()))}")
    except Exception as e:
        print(f"  {trait:18}: Failed to read ({e})")
