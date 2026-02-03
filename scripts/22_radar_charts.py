import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def normalize_data(df, model_col='model', trait_col='trait', value_col='value'):
    """
    Normalizes data for radar charts:
    - Extracts split (base/instruct) from model name or separate column.
    - Normalizes model names for display.
    """
    # Normalize model names: remove _base, _instruct, etc.
    if 'split' not in df.columns:
        df['split'] = df[model_col].apply(lambda x: 'base' if 'base' in x or 'Base' in x else 'instruct')
        
    # Clean model names for legend
    def clean_name(row):
        name = row[model_col]
        # Remove common suffixes/prefixes to get the core model name
        for pat in ['_base', '_instruct', 'Base', 'Instruct', 'mistralai/', 'meta-llama/', 'allenai/', 'Qwen/', 'google/', 'tiiuae/']:
            name = name.replace(pat, '')
        name = name.strip('-_/')
        return name

    df['display_name'] = df.apply(clean_name, axis=1)
    return df

def create_radar_chart(df, categories, title, output_path, value_col):
    """
    Creates a radar chart comparing multiple models on specified categories.
    """
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the circle
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels using the columns
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    
    # Add plots
    models = df['display_name'].unique()
    
    # Define a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        subset = df[df['display_name'] == model]
        
        # Ensure values are sorted by category order
        values = []
        for cat in categories:
            val = subset[subset['trait'] == cat][value_col].values
            if len(val) > 0:
                values.append(val[0])
            else:
                values.append(0) # Handle missing data
                
        values += values[:1] # Close the circle
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    plt.title(title, size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Radar Charts for Personality Vectors")
    parser.add_argument("--internal_glob", required=True, help="Glob for slopes CSVs (Internal Score)")
    parser.add_argument("--external_glob", required=True, help="Glob for text sensitivities CSVs (External Score)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    
    # Model Alpha Ranges (Max absolute value)
    # Used to normalize slopes: Normalized Slope = Raw Slope * Max Alpha
    # This represents the total change over the model's full operating range.
    MAX_ALPHAS = {
        "mistral_7b": 2.0,
        "llama3_8b": 7.0,
        "olmo3_7b": 15.0,
        "qwen25_7b": 50.0,
        "gemma2_9b": 200.0,
        "falcon3_7b": 100.0
    }
    
    # helper to get max alpha
    def get_max_alpha(model_tag):
        # fuzzy match or direct lookup
        for key, val in MAX_ALPHAS.items():
            if key in model_tag:
                return val
        return 1.0 # Default if not found

    # ==================== INTERNAL SCORES (Probe Slopes) ====================
    print("Processing Internal Scores...")
    internal_files = glob.glob(args.internal_glob)
    df_internal_list = []
    
    for f in internal_files:
        try:
            # Check headers to confirm it's a slope file
            # slope file should have: slope_delta_score_vs_alpha
            temp = pd.read_csv(f)
            if 'slope_delta_score_vs_alpha' not in temp.columns:
                continue
                
            # Need to identify model from filename or path if not in csv
            # Path structure: exp/TAG/results...
            path_parts = f.split(os.sep)
            if 'exp' in path_parts:
                idx = path_parts.index('exp')
                if idx + 1 < len(path_parts):
                    model_tag = path_parts[idx+1]
                    temp['model_tag'] = model_tag
            
            df_internal_list.append(temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_internal_list:
        df_int = pd.concat(df_internal_list, ignore_index=True)
        
        if set(df_int['model'].unique()) <= {'base', 'instruct'}:
             df_int['split'] = df_int['model']
             df_int['full_name'] = df_int['model_tag'] + "_" + df_int['split']
        else:
             df_int['split'] = df_int['model'].apply(lambda x: 'base' if 'base' in x else 'instruct')
             df_int['full_name'] = df_int['model_tag'] + "_" + df_int['split']

        # Apply Normalization
        df_int['max_alpha'] = df_int['model_tag'].apply(get_max_alpha)
        # Normalized Slope = Raw Slope * Max Alpha
        df_int['normalized_slope'] = df_int['slope_delta_score_vs_alpha'] * df_int['max_alpha']
        
        # Aggregate: Mean normalized slope across layers
        agg_int = df_int.groupby(['full_name', 'split', 'trait'])['normalized_slope'].mean().reset_index()
        
        # Simplify display names
        agg_int = normalize_data(agg_int, model_col='full_name', value_col='normalized_slope')
        
        # Plot 1: Base Only
        base_df = agg_int[agg_int['split'] == 'base']
        if not base_df.empty:
            create_radar_chart(base_df, TRAITS, "Internal Sensitivity (Base Models)\n(Normalized by Max Range)", 
                               os.path.join(args.out_dir, "radar_internal_base.png"), 'normalized_slope')
            
        # Plot 2: Instruct Only
        instr_df = agg_int[agg_int['split'] == 'instruct']
        if not instr_df.empty:
            create_radar_chart(instr_df, TRAITS, "Internal Sensitivity (Instruct Models)\n(Normalized by Max Range)", 
                               os.path.join(args.out_dir, "radar_internal_instruct.png"), 'normalized_slope')

    # ==================== EXTERNAL SCORES (Text Sensitivity) ====================
    print("Processing External Scores...")
    external_files = glob.glob(args.external_glob)
    df_external_list = []
    
    for f in external_files:
        try:
            temp = pd.read_csv(f)
             # Expected cols: trait, split, score_sensitivity...
            if 'score_sensitivity' not in temp.columns:
                continue
                
            path_parts = f.split(os.sep)
            if 'exp' in path_parts:
                idx = path_parts.index('exp')
                if idx + 1 < len(path_parts):
                    model_tag = path_parts[idx+1]
                    temp['model_tag'] = model_tag
            
            df_external_list.append(temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_external_list:
        df_ext = pd.concat(df_external_list, ignore_index=True)
        
        df_ext['full_name'] = df_ext['model_tag'] + "_" + df_ext['split']
        
        # Apply Normalization
        df_ext['max_alpha'] = df_ext['model_tag'].apply(get_max_alpha)
        # Normalized Sensitivity = Raw Sensitivity * Max Alpha
        df_ext['normalized_sensitivity'] = df_ext['score_sensitivity'] * df_ext['max_alpha']
        
        agg_ext = df_ext.groupby(['full_name', 'split', 'trait'])['normalized_sensitivity'].mean().reset_index()
        
        agg_ext = normalize_data(agg_ext, model_col='full_name', value_col='normalized_sensitivity')
        
        # Plot 3: Base Only
        base_df = agg_ext[agg_ext['split'] == 'base']
        if not base_df.empty:
            create_radar_chart(base_df, TRAITS, "External Sensitivity (Base Models)\n(Normalized by Max Range)", 
                               os.path.join(args.out_dir, "radar_external_base.png"), 'normalized_sensitivity')
            
        # Plot 4: Instruct Only
        instr_df = agg_ext[agg_ext['split'] == 'instruct']
        if not instr_df.empty:
            create_radar_chart(instr_df, TRAITS, "External Sensitivity (Instruct Models)\n(Normalized by Max Range)", 
                               os.path.join(args.out_dir, "radar_external_instruct.png"), 'normalized_sensitivity')

if __name__ == "__main__":
    main()
