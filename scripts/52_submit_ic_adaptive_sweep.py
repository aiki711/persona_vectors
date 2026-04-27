import os
import subprocess
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
VALS = [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]

PBS_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ic_steer_{trait}
#SBATCH --partition=GPU-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=log/ic_steer_{trait}.out
#SBATCH --error=log/ic_steer_{trait}.err

WORKDIR="/home/s2550009/persona_vectors"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/src:$WORKDIR:$WORKDIR/scripts:${{PYTHONPATH:-}}"
PYTHON_BIN="$WORKDIR/persona_steering/bin/python3"

# Loop through layers and values
LAYERS=({layers_str})
VALS=({vals_str})

for L in "${{LAYERS[@]}}"; do
    for V in "${{VALS[@]}}"; do
        echo "Running: Trait={trait}, Layer=$L, Tau=$V"
        "$PYTHON_BIN" scripts/50_run_ic_adaptive_steering.py \\
            --config config/mistral_7b.yaml \\
            --vector_bank exp_steering_layer_sweep_1-40/vectors/mean_diff_vectors.npz \\
            --prompts exp_steering_layer_sweep_1-40/test_prompts_10.jsonl \\
            --out_dir exp_steering_ic_adaptive/results/{trait} \\
            --axis {trait} \\
            --target_layer "$L" \\
            --tau "$V" \\
            --ic_scale 1.5
    done
done
"""

def main():
    job_dir = Path("jobs/ic_adaptive_sweep")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    layers_str = " ".join(map(str, LAYERS))
    vals_str = " ".join(map(str, VALS))
    
    for trait in TRAITS:
        pbs_content = PBS_TEMPLATE.format(
            trait=trait,
            layers_str=layers_str,
            vals_str=vals_str
        )
        
        pbs_file = job_dir / f"run_ic_{trait}.pbs"
        with open(pbs_file, "w") as f:
            f.write(pbs_content)
        
        print(f"Submitting job for {trait}...")
        subprocess.run(["sbatch", str(pbs_file)])

if __name__ == "__main__":
    main()
