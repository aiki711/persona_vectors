#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/01_vectors/37_train_soft_probe_filters.py
#
# Trains Logistic Regression probes on positive calibration hidden states H_pos_1000
# and reconstructed negative hidden states L_neg_1000.
# Extracts normalized continuous soft weights (0.0 to 1.0) as masks.
#

import argparse
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = list(range(32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector_bank",   default="vectors/mean_diff_vectors.npz")
    ap.add_argument("--out_mask_bank", default="vectors/soft_probe_masks.npz")
    args = ap.parse_args()

    # Login node execution guard to prevent server overload
    import socket
    import sys
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] This heavy computation script cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job using sbatch to run it on a compute node.")
        sys.exit(1)

    print("=== 37_train_soft_probe_filters.py ===")
    print(f"  Vector Bank: {args.vector_bank}")
    print(f"  Output Bank: {args.out_mask_bank}")

    # Load mean diff vector bank
    v_data = np.load(args.vector_bank)
    final_masks = {}

    # If the output file already exists, load it to preserve other traits/configurations
    out_path = Path(args.out_mask_bank)
    if out_path.exists():
        try:
            existing = np.load(out_path)
            final_masks.update({k: existing[k] for k in existing.files})
            print(f"Loaded {len(existing.files)} existing masks to preserve.")
        except Exception as e:
            print(f"Warning: failed to load existing mask file: {e}")

    for axis in TRAITS:
        print(f"\nProcessing Axis: {axis}...")
        for L in LAYERS:
            h_pos_key = f"{L}|{axis}|H_pos_1000"
            mp_key = f"{L}|{axis}|midpoint"

            if h_pos_key not in v_data or mp_key not in v_data:
                h_pos_key = f"{L}|{axis}|H_pos_30"
                if h_pos_key not in v_data:
                    continue

            # Load representation arrays
            H_pos = v_data[h_pos_key].astype(np.float32) # Shape: [N, D]
            midpoint = v_data[mp_key].astype(np.float32)  # Shape: [D]

            # Reconstruct negative representations
            L_neg = 2.0 * midpoint.reshape(1, -1) - H_pos

            # Construct dataset
            X = np.concatenate([H_pos, L_neg], axis=0) # [2*N, D]
            y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(L_neg))], axis=0)

            # Standardize features to avoid scale/outlier bias
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit Logistic Regression probe
            clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_scaled, y)

            # Get probe coefficients
            coef = clf.coef_[0]  # [D]
            abs_coef = np.abs(coef)

            # Create continuous soft mask normalized to [0, 1]
            max_val = abs_coef.max()
            soft_mask = abs_coef / (max_val + 1e-10)

            # Save mask
            mask_key = f"{L}|{axis}|mask"
            final_masks[mask_key] = soft_mask

        print(f"  Completed training soft masks for {axis}.")

    # Save to npz file
    np.savez_compressed(args.out_mask_bank, **final_masks)
    print(f"\n[DONE] Saved probe-trained soft dimension masks to: {args.out_mask_bank}")

if __name__ == "__main__":
    main()
