import os
from pathlib import Path
import datetime

def check_mtime(path):
    p = Path(path)
    if p.exists():
        mtime = os.path.getmtime(p)
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"{path}: last modified at {dt}")
    else:
        print(f"{path}: does not exist")

check_mtime("exp_steering_layer_analysis/vectors/mean_diff_vectors.npz")
check_mtime("archive_exp/exp_steering_layer_sweep_5-25/vectors/mean_diff_vectors.npz")
check_mtime("archive_exp/exp_steering_layer_sweep_1-40/vectors/mean_diff_vectors.npz")
check_mtime("archive_exp/exp_adaptive_steering/vectors/mean_diff_vectors.npz")
