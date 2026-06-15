import numpy as np
import torch

def check_npz(path):
    print(f"=== Checking {path} ===")
    try:
        data = np.load(path)
        keys = list(data.keys())
        print("Total keys:", len(keys))
        # Print a few example keys
        sample_keys = [k for k in keys if "extraversion" in k][:15]
        print("Sample keys (extraversion):", sample_keys)
        
        # Check if raw_norm, midpoint, and w exist for layer 10 and 20 extraversion
        for L in [10, 20]:
            w_key = f"{L}|extraversion|w"
            mp_key = f"{L}|extraversion|midpoint"
            rn_key = f"{L}|extraversion|raw_norm"
            
            w_str = f"w: shape={data[w_key].shape}, norm={np.linalg.norm(data[w_key]):.4f}" if w_key in data else "w: Missing"
            mp_str = f"midpoint: shape={data[mp_key].shape}, norm={np.linalg.norm(data[mp_key]):.4f}" if mp_key in data else "midpoint: Missing"
            rn_str = f"raw_norm: {data[rn_key]}" if rn_key in data else "raw_norm: Missing"
            
            print(f"Layer {L}: {w_str} | {mp_str} | {rn_str}")
    except Exception as e:
        print("Error:", e)

check_npz("vectors/mean_diff_vectors.npz")
check_npz("exp_steering_layer_analysis/vectors/mean_diff_vectors.npz")
check_npz("archive_exp/exp_steering_layer_sweep_5-25/vectors/mean_diff_vectors.npz")
