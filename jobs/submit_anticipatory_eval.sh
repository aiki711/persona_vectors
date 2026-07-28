#!/bin/bash
#SBATCH --job-name=anticipat_eval
#SBATCH --output=log/anticipatory_eval.log
#SBATCH --error=log/anticipatory_eval.err
#SBATCH --partition=GPU-1
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=spcc-a40g04

set -e

mkdir -p log
source persona_steering/bin/activate

echo "Starting 70B Evaluation for Anticipatory vs Delayed experiment..."
date

for jsonl in $(find exp_token_intensity/exp_resampling_vs_delayed -name '*.jsonl'); do
    trait=$(basename $(dirname $jsonl))
    echo "Evaluating: $jsonl (Trait: $trait)..."
    python scripts/04_dyn_layer/02_token_intensity/batch_eval.py --file "$jsonl" --axis "$trait" --quant 4bit
done

echo "Generating comparison plots and report..."
python scratch/plot_anticipatory_comparison.py

echo "Evaluation Finished Successfully!"
date
