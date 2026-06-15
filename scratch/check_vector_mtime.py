import os
from pathlib import Path
import datetime

p = Path("vectors/mean_diff_vectors.npz")
if p.exists():
    mtime = os.path.getmtime(p)
    dt = datetime.datetime.fromtimestamp(mtime)
    print(f"vectors/mean_diff_vectors.npz: last modified at {dt}")
else:
    print("vectors/mean_diff_vectors.npz: does not exist")
