import os
import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=eval_vs_const_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=log/eval_vs_const_{trait}.out
#SBATCH --error=log/eval_vs_const_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Evaluating IC-Adaptive pairwise for {trait}..."
"$PYTHON_BIN" scripts/03_ic_adaptive/53_eval_ic_pairwise_all.py --trait {trait}
"""

def main():
    job_dir = Path("jobs/ic_adaptive_eval")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        
        pbs_file = job_dir / f"run_eval_ic_{trait}.pbs"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        
        print(f"Submitting eval job for {trait}...")
        subprocess.run(["sbatch", str(pbs_file)])

if __name__ == "__main__":
    main()
