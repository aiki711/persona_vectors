import json
import torch
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device, get_layer_stack

AXIS_LIST = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]
N_SAMPLES = 30
TEST_PROMPTS_LIMIT = 10
LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
PENALTIES = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.0]

def load_layer_priors(input_dir: Path, axis: str) -> dict:
    trait_dir = input_dir / axis
    priors = {L: 0.0 for L in LAYERS}
    for L in LAYERS:
        max_safe_dev = 0.0
        has_any_data = False
        for val in VALS:
            csv_path = trait_dir / f"scores_layer_{L}_Val{float(val)}.csv"
            if not csv_path.exists():
                csv_path = trait_dir / f"scores_layer_{L}_Val{val}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    mean_score = df["const_score"].mean()
                    mean_ppl = df["const_ppl"].mean()
                    has_any_data = True
                    if mean_ppl <= 25.0:
                        dev = mean_score - 3.0
                        if dev > max_safe_dev:
                            max_safe_dev = dev
                except Exception:
                    pass
        if has_any_data:
            priors[L] = max(0.0, max_safe_dev)
        else:
            if 4 <= L <= 29:
                priors[L] = 1.0
            else:
                priors[L] = 0.0
    for L in priors:
        if not (4 <= L <= 29):
            priors[L] = 0.0
    max_w = max(priors.values())
    if max_w > 1e-8:
        for L in priors:
            priors[L] /= max_w
    return priors

def compute_entropy(selections):
    counts = pd.Series(selections).value_counts()
    probs = counts / len(selections)
    return -np.sum(probs * np.log2(probs))

def main():
    config_path = Path("config/mistral_7b.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_name = cfg.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
    print(f"Loading model: {model_name}...")
    model, tok = load_model_and_tokenizer(model_name, quant="4bit")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    layers_stack, N_layers, _ = get_layer_stack(model)
    layer_indices = list(range(N_layers))
    device = _infer_main_device(model)
    model.eval()
    
    # Load evaluation prompts
    prompts_path = Path("inputs/eval_prompts_10.jsonl")
    if not prompts_path.exists():
        prompts_path = Path("inputs/test_prompts_10.jsonl")
    
    all_prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]"): continue
            if line.endswith(","): line = line[:-1]
            try: item = json.loads(line)
            except: item = line.strip('"')
            if isinstance(item, dict) and "input" in item:
                all_prompts.append(item["input"])
            elif isinstance(item, str):
                all_prompts.append(item)
    
    prompts = all_prompts[:TEST_PROMPTS_LIMIT]

    print(f"Pre-computing hidden states for {len(prompts)} test prompts...")
    prompts_h = []
    for p_idx, p_text in enumerate(prompts):
        formatted = tok.apply_chat_template([{"role": "user", "content": p_text}], add_generation_prompt=True, tokenize=True)
        input_ids = torch.tensor([formatted]).to(device)
        
        saved_h = {}
        handles = []
        def get_hook(L):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                saved_h[L] = hs[0, -1, :].detach().cpu().float().numpy()
            return hook
            
        for L in layer_indices:
            handles.append(layers_stack[L].register_forward_hook(get_hook(L)))
            
        try:
            with torch.no_grad():
                _ = model(input_ids)
        finally:
            for h in handles:
                h.remove()
        
        prompt_h_dict = {}
        for L in layer_indices:
            h_input = saved_h[L]
            h_unit = h_input / (np.linalg.norm(h_input) + 1e-10)
            prompt_h_dict[L] = h_unit
        prompts_h.append(prompt_h_dict)

    input_dir = Path("exp_steering_layer_analysis/results")
    v_data = np.load("vectors/mean_diff_vectors.npz")
    
    out_img_dir = Path("log/figures_rank_selection")
    out_img_dir.mkdir(parents=True, exist_ok=True)

    summary_entropy_data = []

    for axis in AXIS_LIST:
        print(f"\nProcessing {axis}...")
        layer_priors = load_layer_priors(input_dir, axis)
        
        layer_w = {}
        h_pos_dict = {}
        h_pos_all = {L: [] for L in layer_indices}
        for L in layer_indices:
            w_key = f"{L}|{axis}|w"
            if w_key in v_data:
                layer_w[L] = v_data[w_key]
            
            H_pos_key = f"{L}|{axis}|H_pos_30"
            h_pos_key = f"{L}|{axis}|h_pos_30"
            if H_pos_key in v_data:
                H_pos = v_data[H_pos_key]
                h_pos_dict[L] = v_data[h_pos_key]
                h_pos_all[L].append(H_pos)

        cosine_prior_selections = {p: [] for p in PENALTIES}
        rank_selections = {p: [] for p in PENALTIES}
        
        for p_idx, p_text in enumerate(prompts):
            h_dict = prompts_h[p_idx]
            
            cos_scores = {}
            percentiles = {}
            
            for L in layer_indices:
                h_unit = h_dict[L]
                if L in layer_w:
                    w_vec = layer_w[L]
                    w_unit = w_vec / (np.linalg.norm(w_vec) + 1e-10)
                    cos_scores[L] = np.dot(h_unit, w_unit)
                else:
                    cos_scores[L] = -1.0
                
                if L in h_pos_dict:
                    H_pos = np.concatenate(h_pos_all[L], axis=0)
                    h_pos = h_pos_dict[L]
                    Hh_pos = np.concatenate([H_pos, h_pos], axis=0)
                    
                    Hh_pos_norm = Hh_pos / (np.linalg.norm(Hh_pos, axis=1, keepdims=True) + 1e-10)
                    sims_combined = np.dot(Hh_pos_norm, h_unit.T) 
                    
                    sim_input = np.dot(h_unit, (h_pos / np.linalg.norm(h_pos)).T).item()
                    sims_with_input = np.concatenate([sims_combined, [sim_input]])
                    ranking_with_input = np.argsort(sims_with_input)
                    
                    rank_idx = np.where(ranking_with_input == len(sims_with_input) - 1)[0][0]
                    percentile = rank_idx / float(len(sims_combined))
                    percentiles[L] = percentile
                else:
                    percentiles[L] = 999.0

            for penalty in PENALTIES:
                cos_prior_scores = {}
                rank_scores = {}
                for L in layer_indices:
                    w_prior = layer_priors.get(L, 0.0)
                    if w_prior > 1e-5:
                        cos_prior_scores[L] = cos_scores[L] - (1.0 - w_prior) * penalty
                        rank_scores[L] = percentiles[L] + (1.0 - w_prior) * penalty
                    else:
                        cos_prior_scores[L] = -999.0
                        rank_scores[L] = 999.0
                
                best_cos_prior = max(cos_prior_scores, key=lambda L: cos_prior_scores[L])
                best_rank = min(rank_scores, key=lambda L: rank_scores[L])
                
                cosine_prior_selections[penalty].append(best_cos_prior)
                rank_selections[penalty].append(best_rank)

        # Let's plot unified figure for this axis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Heatmap for Cosine-Prior
        cos_matrix = np.array([cosine_prior_selections[p] for p in PENALTIES])
        sns.heatmap(cos_matrix, annot=True, cmap="YlOrRd", fmt="d",
                    xticklabels=[f"P{i+1}" for i in range(TEST_PROMPTS_LIMIT)],
                    yticklabels=[f"P={p}" for p in PENALTIES],
                    ax=axes[0, 0], cbar=False)
        axes[0, 0].set_title(f"Cosine-Prior Selected Layers [{axis.capitalize()}]")
        axes[0, 0].set_xlabel("Evaluation Prompts")
        axes[0, 0].set_ylabel("Penalty Coefficient")
        
        # 2. Heatmap for Rank-based
        rank_matrix = np.array([rank_selections[p] for p in PENALTIES])
        sns.heatmap(rank_matrix, annot=True, cmap="YlOrRd", fmt="d",
                    xticklabels=[f"P{i+1}" for i in range(TEST_PROMPTS_LIMIT)],
                    yticklabels=[f"P={p}" for p in PENALTIES],
                    ax=axes[0, 1], cbar=False)
        axes[0, 1].set_title(f"Rank-based Selected Layers [{axis.capitalize()}]")
        axes[0, 1].set_xlabel("Evaluation Prompts")
        axes[0, 1].set_ylabel("Penalty Coefficient")

        # 3. Entropy Line Chart
        cos_entropies = [compute_entropy(cosine_prior_selections[p]) for p in PENALTIES]
        rank_entropies = [compute_entropy(rank_selections[p]) for p in PENALTIES]
        
        axes[1, 0].plot(PENALTIES, cos_entropies, marker='o', color='red', label='Cosine-Prior')
        axes[1, 0].plot(PENALTIES, rank_entropies, marker='s', color='blue', label='Rank-based')
        axes[1, 0].set_xscale('symlog', linthresh=0.1)
        axes[1, 0].invert_xaxis()
        axes[1, 0].set_title(f"Selection Entropy vs Penalty Coefficient [{axis.capitalize()}]")
        axes[1, 0].set_xlabel("Penalty Coefficient")
        axes[1, 0].set_ylabel("Shannon Entropy (bits)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)

        # 4. Safe prior weight of selected layers distribution
        # Compute mean safety prior weight of selected layers for each penalty
        rank_mean_priors = []
        cos_mean_priors = []
        for p in PENALTIES:
            rank_mean_priors.append(np.mean([layer_priors.get(L, 0.0) for L in rank_selections[p]]))
            cos_mean_priors.append(np.mean([layer_priors.get(L, 0.0) for L in cosine_prior_selections[p]]))
            summary_entropy_data.append({
                "axis": axis,
                "penalty": p,
                "cos_entropy": cos_entropies[PENALTIES.index(p)],
                "rank_entropy": rank_entropies[PENALTIES.index(p)],
                "cos_mean_safety": cos_mean_priors[-1],
                "rank_mean_safety": rank_mean_priors[-1]
            })

        axes[1, 1].plot(PENALTIES, cos_mean_priors, marker='o', color='red', linestyle='--', label='Cosine-Prior Safety')
        axes[1, 1].plot(PENALTIES, rank_mean_priors, marker='s', color='blue', linestyle='--', label='Rank-based Safety')
        axes[1, 1].set_xscale('symlog', linthresh=0.1)
        axes[1, 1].invert_xaxis()
        axes[1, 1].set_title(f"Mean Safety Prior Weight vs Penalty [{axis.capitalize()}]")
        axes[1, 1].set_xlabel("Penalty Coefficient")
        axes[1, 1].set_ylabel("Safety Prior Weight")
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)

        plt.suptitle(f"Dynamic Layer Selection Analysis: {axis.capitalize()}", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        out_path = out_img_dir / f"heatmap_{axis}_selection_analysis.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved figure for {axis} to {out_path}")

    # Summary plot combining all traits
    df_summary = pd.DataFrame(summary_entropy_data)
    plt.figure(figsize=(10, 6))
    for axis in AXIS_LIST:
        df_sub = df_summary[df_summary["axis"] == axis]
        plt.plot(df_sub["penalty"], df_sub["rank_entropy"], marker='o', label=axis.capitalize())
    plt.xscale('symlog', linthresh=0.1)
    plt.gca().invert_xaxis()
    plt.title("Rank-Based DLS Selection Entropy vs Penalty Coefficient (All Traits)")
    plt.xlabel("Penalty Coefficient")
    plt.ylabel("Selection Entropy (bits)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    summary_path = out_img_dir / "summary_rank_entropy_all_traits.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure to {summary_path}")

if __name__ == "__main__":
    main()
