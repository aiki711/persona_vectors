import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import json
from tqdm import tqdm

def get_color(alpha):
    if alpha > 0:
        return 'red'
    elif alpha < 0:
        return 'blue'
    else:
        return 'gray'

def plot_scatter(x, y, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(6, 5))
    
    # 色のリスト作成
    try:
        colors = [get_color(a) for a in x]
    except Exception:
        colors = 'blue'

    # 散布図
    plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='k', linewidth=0.3)
    
    # グリッドと基準線
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()

def process_external_scores(score_files, out_dir):
    """
    Given a list of score CSVs, generate scatter plots.
    """
    print(f"[External] Processing {len(score_files)} score CSVs.")
    
    plot_dir = os.path.join(out_dir, "external")
    os.makedirs(plot_dir, exist_ok=True)

    for f in score_files:
        try:
            df = pd.read_csv(f)
            
            # --- カラム名の揺らぎ吸収 ---
            alpha_col = None
            if 'alpha' in df.columns:
                alpha_col = 'alpha'
            elif 'alpha_total' in df.columns:
                alpha_col = 'alpha_total'
            
            label_mapping = {
                'score_LABEL_0': 'score_extraversion',
                'score_LABEL_1': 'score_neuroticism',
                'score_LABEL_2': 'score_agreeableness',
                'score_LABEL_3': 'score_conscientiousness',
                'score_LABEL_4': 'score_openness'
            }
            df.rename(columns=label_mapping, inplace=True)

            score_cols_found = [c for c in df.columns if c.startswith('score_')]
            
            if alpha_col is None or not score_cols_found:
                if 'score' not in df.columns and 'probability' not in df.columns:
                    # columns check failed
                    print(f"[Skip] {os.path.basename(f)}: Missing columns.")
                    continue
            
            if not score_cols_found:
                if 'score' in df.columns:
                    score_cols_found = ['score']
                elif 'probability' in df.columns:
                    df['score'] = df['probability']
                    score_cols_found = ['score']

            # --- Meta Info ---
            basename = os.path.basename(f)
            # Try to guess model/split from filename or path if possible, 
            # but relying on filename is safer if path structure varies.
            # Filename format expected: {tag}_{split}_personality_scores.csv 
            # OR {tag}_{split}_{trait}_personality_scores.csv
            
            name_parts = basename.replace("_personality_scores.csv", "").split('_')
            # Heuristic: "base" or "instruct" usually indicate the split location
            if "base" in name_parts:
                split = "base"
            elif "instruct" in name_parts:
                split = "instruct"
            else:
                split = "unknown"
                
            model_tag = name_parts[0] # Very rough guess

            # --- Plotting ---
            if 'trait' in df.columns:
                traits = df['trait'].unique()
                for t in traits:
                    subset = df[df['trait'] == t]
                    if subset.empty: continue
                    
                    target_col = None
                    for c in score_cols_found:
                        if t.lower() in c.lower():
                            target_col = c
                            break
                    if target_col is None and 'score' in df.columns:
                        target_col = 'score'
                        
                    if target_col is None: continue

                    out_name = f"{basename.replace('.csv', '')}_{t}.png"
                    plot_scatter(
                        subset[alpha_col], subset[target_col], 
                        f"External Score: {t}\n({model_tag} / {split})",
                        "Alpha", f"BERT Score ({target_col})",
                        os.path.join(plot_dir, out_name)
                    )
            else:
                for col in score_cols_found:
                    trait_name = col.replace("score_", "")
                    out_name = f"{basename.replace('.csv', '')}_{trait_name}.png"
                    plot_scatter(
                        df[alpha_col], df[col], 
                        f"External Score: {trait_name}\n({model_tag} / {split})",
                        "Alpha", f"BERT Score ({trait_name})",
                        os.path.join(plot_dir, out_name)
                    )

        except Exception as e:
            print(f"[Error] Processing {f}: {e}")

def process_internal_sensitivity(metrics_files, out_dir):
    """
    The original code looked for jsonl files to plot internal sensitivity (cosine sim).
    But the new args provide metrics CSV files (text_metrics).
    
    If the user intention is to plot 'Cosine Similarity' or 'Perplexity' vs Alpha from text_metrics.csv,
    we can do that here.
    
    However, the request mentioned "Internal" scatter plots which originally came from _probe_results.jsonl's s_avg.
    The run_neutral_experiments.sh passes:
      --metrics_glob "${results_dir}/*_text_metrics.csv"
      --score_glob "${results_dir}/*_personality_scores.csv"
    
    It does NOT pass jsonl files. So we should use the text_metrics.csv here.
    text_metrics.csv usually contains: alpha, edit_distance, perplexity, etc.
    """
    print(f"[Internal] Processing {len(metrics_files)} metrics CSVs.")
    
    plot_dir = os.path.join(out_dir, "internal")
    os.makedirs(plot_dir, exist_ok=True)
    
    for f in metrics_files:
        try:
            df = pd.read_csv(f)
            
            # --- Identify Alpha Column ---
            alpha_col = None
            if 'alpha' in df.columns:
                alpha_col = 'alpha'
            elif 'alpha_total' in df.columns:
                alpha_col = 'alpha_total'
            
            if alpha_col is None:
                print(f"[Skip] {os.path.basename(f)}: Missing alpha column.")
                continue
            
            # --- Decide what to plot ---
            cols_to_plot = []
            
            # Edit Distance variants
            if 'normalized_distance' in df.columns:
                cols_to_plot.append(('normalized_distance', 'Normalized Edit Distance'))
            elif 'levenshtein_distance' in df.columns:
                cols_to_plot.append(('levenshtein_distance', 'Levenshtein Distance'))
            elif 'edit_distance' in df.columns:
                cols_to_plot.append(('edit_distance', 'Edit Distance'))
                
            # Perplexity variants
            if 'perplexity' in df.columns:
                cols_to_plot.append(('perplexity', 'Perplexity'))
            elif 'ppl' in df.columns:
                cols_to_plot.append(('ppl', 'Perplexity'))
            
            basename = os.path.basename(f).replace("_text_metrics.csv", "")
            
            for col, label in cols_to_plot:
                out_name = f"{basename}_{col}.png"
                plot_scatter(
                    df[alpha_col], df[col],
                    f"{label} vs Alpha\n({basename})",
                    "Alpha", label,
                    os.path.join(plot_dir, out_name)
                )
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

def process_internal_states_from_jsonl(jsonl_files, out_dir):
    """
    Process probe results JSONL to plot internal cosine similarity (s_avg) vs alpha.
    """
    print(f"[Internal] Processing {len(jsonl_files)} JSONL files for cosine similarity.")
    
    plot_dir = os.path.join(out_dir, "internal")
    # os.makedirs(plot_dir, exist_ok=True) # Already created or shared

    for f in jsonl_files:
        try:
            # We only need alpha and s_avg. 
            # Reading entire file into pandas might be slow if huge, but typically manageable.
            data = []
            with open(f, 'r') as fh:
                for line in fh:
                    if not line.strip(): continue
                    try:
                        row = json.loads(line)
                        val_alpha = row.get('alpha')
                        if val_alpha is None:
                            val_alpha = row.get('alpha_total')
                        
                        val_savg = row.get('s_avg')
                        
                        if val_alpha is not None and val_savg is not None:
                            data.append({'alpha': val_alpha, 's_avg': val_savg})
                    except Exception:
                        pass
            
            if not data:
                continue
                
            df = pd.DataFrame(data)
            
            basename = os.path.basename(f).replace(".jsonl", "")
            # Filter name if it's too long or has redundant info? 
            # e.g. mistral_7b_base_agreeableness_probe_results -> agreeableness
            
            # Simple plot
            out_name = f"{basename}_cosine_sim.png"
            plot_scatter(
                df['alpha'], df['s_avg'],
                f"Internal Cosine Similarity vs Alpha\n({basename})",
                "Alpha", "Cosine Similarity (s_avg)",
                os.path.join(plot_dir, out_name)
            )

        except Exception as e:
            print(f"[Error] Processing JSONL {f}: {e}")

def main():
    parser = argparse.ArgumentParser()
    # New Arguments style
    parser.add_argument("--metrics_glob", type=str, default=None, help="Glob pattern for text metrics CSVs")
    parser.add_argument("--score_glob", type=str, default=None, help="Glob pattern for personality scores CSVs")
    parser.add_argument("--jsonl_glob", type=str, default=None, help="Glob pattern for probe results JSONL files (for internal similarity)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--tag", type=str, default="scatter", help="Tag for the run")
    
    # Legacy arguments (kept for compatibility if needed, though we primarily use new ones)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--suffix", default=None)
    
    args = parser.parse_args()
    
    # Resolve files
    score_files = []
    if args.score_glob:
        score_files = glob.glob(args.score_glob)
    
    metrics_files = []
    if args.metrics_glob:
        metrics_files = glob.glob(args.metrics_glob)
        
    jsonl_files = []
    if args.jsonl_glob:
        jsonl_files = glob.glob(args.jsonl_glob)
       
    # If using legacy mode (fallback)
    if not score_files and not metrics_files and not jsonl_files and args.root_dir:
        pass

    if score_files:
        process_external_scores(score_files, args.out_dir)
    
    if metrics_files:
        process_internal_sensitivity(metrics_files, args.out_dir)

    if jsonl_files:
        process_internal_states_from_jsonl(jsonl_files, args.out_dir)

    print("Done.")

if __name__ == "__main__":
    main()