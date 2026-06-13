import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=re_eval_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --output=log/re_eval_{trait}.out
#SBATCH --error=log/re_eval_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

source persona_steering/bin/activate 2>/dev/null || conda activate "$WORKDIR/persona_steering" 2>/dev/null || true
export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

echo "Starting re-evaluation of baselines for {trait}..."
"$PYTHON_BIN" scratch/batch_re_eval.py --axis {trait}
echo "Done."
"""

def main():
    job_dir = Path("jobs/re_eval_baselines")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(trait=trait)
        pbs_file = job_dir / f"run_re_eval_{trait}.sh"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        pbs_file.chmod(0o755)
        
        cmd = ["sbatch", str(pbs_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Submitted re-eval job for {trait}: {res.stdout.strip()}")

if __name__ == "__main__":
    main()
