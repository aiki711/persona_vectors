import numpy as np

def compare_npz(path1, path2):
    print(f"=== Comparing {path1} and {path2} ===")
    try:
        d1 = np.load(path1)
        d2 = np.load(path2)
        
        # Check L20 extraversion w
        k_w = "20|extraversion|w"
        k_m = "20|extraversion|midpoint"
        
        if k_w in d1 and k_w in d2:
            diff_w = np.linalg.norm(d1[k_w] - d2[k_w])
            print(f"w diff (Layer 20): {diff_w:.8f}")
        else:
            print(f"w missing in one of the files: in 1: {k_w in d1}, in 2: {k_w in d2}")
            
        if k_m in d1 and k_m in d2:
            diff_m = np.linalg.norm(d1[k_m] - d2[k_m])
            print(f"midpoint diff (Layer 20): {diff_m:.8f}")
        else:
            print(f"midpoint missing in one of the files: in 1: {k_m in d1}, in 2: {k_m in d2}")
    except Exception as e:
        print("Error:", e)

compare_npz("vectors/mean_diff_vectors.npz", "exp_steering_layer_analysis/vectors/mean_diff_vectors.npz")
