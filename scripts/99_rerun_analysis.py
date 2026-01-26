import os
import subprocess
import glob
import shutil

# Configuration matches run_other_alpha_sweep.sh
models = [
    "mistral_7b",
    "llama3_8b",
    "olmo3_7b",
    "qwen25_7b",
    "gemma2_9b",
    "falcon3_7b"
]
splits = ["base", "instruct"]
root_dir = "exp"
python_bin = "python3" # Assuming venv is active or using system python in compatible env

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def process_model(tag):
    print(f"=== Processing {tag} ===")
    results_dir = f"{root_dir}/{tag}/asst_pairwise_results"
    
    # 1. Text Analysis (13 & 14) for both splits
    for split in splits:
        input_jsonl = f"{results_dir}/{tag}_{split}_alltraits.jsonl"
        if not os.path.exists(input_jsonl):
            print(f"Skipping {split}: {input_jsonl} not found.")
            continue

        # Paths
        metrics_csv = f"{results_dir}/{tag}_{split}_text_metrics.csv"
        score_csv = f"{results_dir}/{tag}_{split}_personality_scores.csv"

        # Delete old files
        if os.path.exists(metrics_csv):
            print(f"Deleting old {metrics_csv}")
            os.remove(metrics_csv)
        if os.path.exists(score_csv):
            print(f"Deleting old {score_csv}")
            os.remove(score_csv)

        # Run 13 (New Metrics Script)
        cmd_13 = f"{python_bin} scripts/13_text_metrics_vs_alpha.py '{input_jsonl}' --output '{metrics_csv}' --baseline 0"
        run_cmd(cmd_13)

        # Run 14
        cmd_14 = f"{python_bin} scripts/14_calc_personality_score.py '{input_jsonl}' --output '{score_csv}' --batch_size 32 --model 'Minej/bert-base-personality'"
        run_cmd(cmd_14)

    # 2. Slopes (02 & 03)
    slopes_dir = f"{results_dir}/slopes"
    figs_dir = f"{slopes_dir}/figs"
    os.makedirs(slopes_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    base_all = f"{results_dir}/{tag}_base_alltraits.jsonl"
    instr_all = f"{results_dir}/{tag}_instruct_alltraits.jsonl"
    out_csv = f"{slopes_dir}/slopes_{tag}_asst_pairwise.csv"

    if os.path.exists(base_all) and os.path.exists(instr_all):
        if os.path.exists(out_csv):
            print(f"Deleting old slopes csv: {out_csv}")
            os.remove(out_csv)
        
        cmd_02 = f"{python_bin} scripts/02_probe_slopes_from_logs.py --base_json '{base_all}' --instr_json '{instr_all}' --out_csv '{out_csv}' --pooling asst --axis_mode pairwise"
        run_cmd(cmd_02)

        # 03 Visualize Slopes (optional update)
        run_cmd(f"{python_bin} scripts/03_slopes_visualize.py --input_dir '{out_csv}' --out_dir '{figs_dir}' --value_col slope_delta_score_vs_alpha01")
        run_cmd(f"{python_bin} scripts/03_slopes_visualize.py --input_dir '{out_csv}' --out_dir '{figs_dir}' --value_col slope_delta_score_vs_alpha")

    # 3. Individual Viz (15 & 16)
    out_plots_dir = f"{results_dir}/plots"
    os.makedirs(out_plots_dir, exist_ok=True)

    # 15 (Update to use metrics_glob)
    cmd_15 = f"{python_bin} scripts/15_text_sensitivity_visualize.py --metrics_glob '{results_dir}/*_text_metrics.csv' --score_glob '{results_dir}/*_personality_scores.csv' --out_dir '{out_plots_dir}' --tag '{tag}'"
    run_cmd(cmd_15)

    # 16
    slopes_csv_path = out_csv 
    # note: 16 uses slopes csv and results from 15 (which produces {tag}_text_sensitivities.csv in out_dir)
    sens_csv = f"{out_plots_dir}/{tag}_text_sensitivities.csv"
    
    cmd_16 = f"{python_bin} scripts/16_visualize_combined_metrics.py --internal_csv '{slopes_csv_path}' --external_csv '{sens_csv}' --out_dir '{out_plots_dir}' --tag '{tag}'"
    run_cmd(cmd_16)


def main():
    for model in models:
        try:
            process_model(model)
        except Exception as e:
            print(f"Error processing {model}: {e}")

    print("=== Global Viz ===")
    
    # 17 Cross Model
    cmd_17 = f"{python_bin} scripts/17_cross_model_comparison.py --root_dir '{root_dir}' --out_dir '{root_dir}/_all/comparison_plots'"
    run_cmd(cmd_17)

    # 18 Scatter
    cmd_18 = f"{python_bin} scripts/18_visualize_scatter.py --root_dir '{root_dir}' --suffix ''"
    run_cmd(cmd_18)

if __name__ == "__main__":
    main()
