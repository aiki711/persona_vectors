import json
import numpy as np
from pathlib import Path
from collections import Counter

traits = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
new_results_dir = Path("exp_steering_dyn_layer_proj_prior/results")

def calc_entropy_for_file(filepath):
    if not filepath.exists():
        return 0.0, []
    layers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if "dyn_layer" in row:
                layers.append(row["dyn_layer"])
    if not layers:
        return 0.0, []
    counts = Counter(layers)
    total = len(layers)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs)
    return entropy, layers

print("=== Shannon Entropy for rank_only (alpha=5.0) ===")
for trait in traits:
    filepath = new_results_dir / trait / "rank_only_Val5.0.jsonl"
    entropy, layers = calc_entropy_for_file(filepath)
    print(f"| **{trait.capitalize()}** | {entropy:.4f} | `{layers}` |")
