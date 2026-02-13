import torch
import sys
import os

import numpy as np

def check_vector(path):
    print(f"Checking vector: {path}")
    if not os.path.exists(path):
        print("File does not exist!")
        return
    
    try:
        if path.endswith('.pt'):
            vec = torch.load(path)
            print(f"Type: {type(vec)}")
            if isinstance(vec, dict):
                print(f"Keys: {vec.keys()}")
                for k, v in vec.items():
                    if isinstance(v, torch.Tensor):
                        print(f"Key '{k}': Shape={v.shape}, Norm={v.norm().item():.4f}, Mean={v.mean().item():.4f}, Std={v.std().item():.4f}")
            elif isinstance(vec, torch.Tensor):
                 print(f"Shape={vec.shape}, Norm={vec.norm().item():.4f}, Mean={vec.mean().item():.4f}, Std={vec.std().item():.4f}")
        elif path.endswith('.npz'):
            data = np.load(path)
            print(f"Type: .npz archive")
            print(f"Keys: {list(data.keys())}")
            for k in data.files:
                v = data[k]
                v_t = torch.tensor(v)
                print(f"Key '{k}': Shape={v.shape}, Norm={v_t.norm().item():.4f}, Mean={v_t.mean().item():.4f}, Std={v_t.std().item():.4f}")
                
    except Exception as e:
        print(f"Error loading vector: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_vector.py <path>")
    else:
        check_vector(sys.argv[1])
