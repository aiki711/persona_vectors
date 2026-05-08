#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

MAPPING = {
    # 01_vectors
    "30_train_boundary.py": "01_vectors",
    "30b_train_mean_diff.py": "01_vectors",
    "31_visualize_boundary.py": "01_vectors",
    "check_vector.py": "01_vectors",

    # 02_base_steering
    "32_run_adaptive_steering.py": "02_base_steering",
    "32b_run_full_layer_steering.py": "02_base_steering",
    "33_eval_adaptive_steering.py": "02_base_steering",
    "34_plot_adaptive_tradeoff.py": "02_base_steering",
    "35_plot_comparison.py": "02_base_steering",
    "35_run_all_traits.sh": "02_base_steering",
    "36_adaptive_vocabulary_scan.py": "02_base_steering",
    "37_run_multi_layer_bidirectional.sh": "02_base_steering",
    "38_run_trait_specific_steering.sh": "02_base_steering",
    "39_plot_full_layer_comparison.py": "02_base_steering",
    "40_run_layer_sweep.py": "02_base_steering",
    "40_submit_layer_sweep_parallel.py": "02_base_steering",
    "41_plot_layer_sweep.py": "02_base_steering",
    "43_plot_delta_analysis.py": "02_base_steering",
    "44_eval_pairwise_comparison.py": "02_base_steering",
    "45_plot_pairwise_analysis.py": "02_base_steering",
    "46_plot_ppl_comparison.py": "02_base_steering",
    "47_plot_variance_analysis.py": "02_base_steering",
    "48_rank_layer_efficiency.py": "02_base_steering",
    "49_visualize_token_alpha.py": "02_base_steering",

    # 03_ic_adaptive
    "50_run_ic_adaptive_steering.py": "03_ic_adaptive",
    "51_eval_ic_adaptive_pilot.py": "03_ic_adaptive",
    "52_submit_ic_adaptive_sweep.py": "03_ic_adaptive",
    "52b_submit_ic_abs_sweep.py": "03_ic_adaptive",
    "53_eval_ic_pairwise_all.py": "03_ic_adaptive",
    "53_submit_eval_ic_pairwise.py": "03_ic_adaptive",
    "54_plot_ic_pairwise_analysis.py": "03_ic_adaptive",
    "55_plot_ic_ppl_comparison.py": "03_ic_adaptive",
    "56_plot_ic_delta_analysis.py": "03_ic_adaptive",
    "57_plot_ic_vs_const_side_by_side.py": "03_ic_adaptive",
    "58_visualize_ic_token_alpha.py": "03_ic_adaptive",
    "60_eval_ic_absolute.py": "03_ic_adaptive",

    # 04_dyn_layer
    "59_eval_dynamic_layer.py": "04_dyn_layer",
    "59_run_dynamic_layer_steering.py": "04_dyn_layer",
    "59_submit_dyn_sweep.py": "04_dyn_layer",
    "61_run_dyn_layer_compare.py": "04_dyn_layer",
    "61_submit_dyn_compare.py": "04_dyn_layer",
    "62_eval_dyn_compare.py": "04_dyn_layer",
    "63_plot_dyn_layer_comparison.py": "04_dyn_layer",

    # 99_utils
    "create_ipip_dataset.py": "99_utils",
    "debug_low_scores.py": "99_utils",
    "re_evaluate_pairwise_sweep.py": "99_utils",
    "re_evaluate_sweep.py": "99_utils",
    "test_classifier.py": "99_utils",
    "util_create_neutral_prompts.py": "99_utils",
    "util_extract_prompts.py": "99_utils",
    "util_print_adv_stats.py": "99_utils",
    "util_print_stats.py": "99_utils",
    "util_sample_10_prompts.py": "99_utils",
    "util_summarize_vocab_evolution.py": "99_utils",
    "verify_kevsun_failure.py": "99_utils",
    "verify_llama3_sensitivity.py": "99_utils",
    "visualize_granular_sweep.py": "99_utils",
    "move_internal_states.sh": "99_utils",
    "plot_all_ic.sh": "99_utils",
    
    # archive
    "live_axes_and_hook.py": "archive"
}

def replace_in_files(file_extensions=[".py", ".sh", ".pbs"], search_dirs=["scripts", "jobs"]):
    for s_dir in search_dirs:
        for root, _, files in os.walk(s_dir):
            for file in files:
                if not any(file.endswith(ext) for ext in file_extensions):
                    continue
                file_path = Path(root) / file
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue

                new_content = content
                
                # Replace 'scripts/old_name.py' with 'scripts/new_dir/old_name.py'
                for old_name, new_dir in MAPPING.items():
                    old_path_str1 = f"scripts/{old_name}"
                    new_path_str1 = f"scripts/{new_dir}/{old_name}"
                    
                    # Some scripts might just call the python script directly if they cd into scripts/
                    old_path_str2 = f" {old_name}"
                    new_path_str2 = f" {new_dir}/{old_name}"
                    
                    new_content = new_content.replace(old_path_str1, new_path_str1)
                    # We have to be careful with old_path_str2, only replace if preceded by space and not containing 'scripts/'
                    # Better to just stick to scripts/old_name.py as all PBS and python subprocess use scripts/...

                if content != new_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated paths in {file_path}")

def move_files():
    # Create directories
    for new_dir in set(MAPPING.values()):
        Path(f"scripts/{new_dir}").mkdir(parents=True, exist_ok=True)
    
    # Move files using git mv
    for old_name, new_dir in MAPPING.items():
        old_path = Path(f"scripts/{old_name}")
        new_path = Path(f"scripts/{new_dir}/{old_name}")
        
        if old_path.exists():
            subprocess.run(["git", "mv", str(old_path), str(new_path)], check=False)
            print(f"Moved {old_path} -> {new_path}")
            
if __name__ == "__main__":
    print("Replacing internal paths...")
    replace_in_files()
    print("Moving files...")
    move_files()
    print("Done!")
