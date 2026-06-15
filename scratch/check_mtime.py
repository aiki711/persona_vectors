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

check_mtime("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/logit_diff_Val1.0.jsonl")
check_mtime("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/cos_only_Val1.0.jsonl")
check_mtime("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/rank_only_Val1.0.jsonl")
check_mtime("exp_steering_dyn_layer_proj_prior/results/extraversion/cos_only_Val1.0.jsonl")
check_mtime("exp_steering_dyn_layer_proj_prior/results/extraversion/rank_only_Val1.0.jsonl")
