#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# scripts/01_vectors/36_train_probe_filters.py
#
# Trains Logistic Regression probes on positive calibration hidden states H_pos_1000
# and reconstructed negative hidden states L_neg_1000.
# Extracts the top-K dimensions with the largest probe coefficients as masks.
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
    ap.add_argument("--out_mask_bank", default="vectors/probe_masks.npz")
    ap.add_argument("--k",             type=int, default=250, help="Number of top dimensions to keep")
    args = ap.parse_args()

    # Login node execution guard to prevent server overload
    import socket
    import sys
    hostname = socket.gethostname()
    if "hakusan" in hostname:
        print(f"\n[ERROR] This heavy computation script cannot be run directly on the login node '{hostname}'.")
        print("Please submit this script as a SLURM job using sbatch to run it on a compute node.")
        sys.exit(1)

    print("=== 36_train_probe_filters.py ===")
    print(f"  Vector Bank: {args.vector_bank}")
    print(f"  Output Bank: {args.out_mask_bank}")
    print(f"  Top K dims : {args.k}")

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
                # Fallback to H_pos_30 if 1000 samples are missing
                h_pos_key = f"{L}|{axis}|H_pos_30"
                if h_pos_key not in v_data:
                    continue

            # Load representation arrays
            H_pos = v_data[h_pos_key].astype(np.float32) # Shape: [N, D]
            midpoint = v_data[mp_key].astype(np.float32)  # Shape: [D]

            # Reconstruct negative representations
            # midpoint = (H_pos + L_neg) / 2  =>  L_neg = 2 * midpoint - H_pos
            L_neg = 2.0 * midpoint.reshape(1, -1) - H_pos

            # Construct dataset
            X = np.concatenate([H_pos, L_neg], axis=0) # [2*N, D]
            y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(L_neg))], axis=0)

            # Standardize features to avoid scale/outlier bias
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit Logistic Regression probe
            # Using L2 penalty and liblinear/saga solver
            clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_scaled, y)

            # Get probe coefficients
            coef = clf.coef_[0]  # [D]
            abs_coef = np.abs(coef)

            # Extract top K dimensions
            top_k_indices = np.argsort(abs_coef)[-args.k:]
            
            # Create boolean mask
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_indices] = True

            # Save mask
            mask_key = f"{L}|{axis}|mask"
            final_masks[mask_key] = mask

        print(f"  Completed training masks for {axis}.")

    # Save to npz file
    np.savez_compressed(args.out_mask_bank, **final_masks)
    print(f"\n[DONE] Saved probe-trained dimension masks to: {args.out_mask_bank}")

if __name__ == "__main__":
    main()
