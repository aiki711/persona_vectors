import json
import torch
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from persona_vectors.live_axes import load_model_and_tokenizer, _infer_main_device, get_layer_stack

AXIS = "extraversion"
N_SAMPLES = 30
TEST_PROMPTS_LIMIT = 10
LAYERS = list(range(32))
VALS = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

def extract_positive_texts(axis, limit=30):
    print("Loading Big5Chat dataset...")
    ds_all = load_dataset("wenkai-li/big5_chat")
    split_name = next(iter(ds_all.keys()))
    ds = ds_all[split_name]
    
    texts = []
    for ex in ds:
        tr = (ex.get("trait") or "").strip().lower()
        lv = (ex.get("level") or "").strip().lower()
        if tr == axis and lv == "high":
            to = (ex.get("train_output") or "").strip()
            if to:
                texts.append(to)
                if len(texts) >= limit:
                    break
    print(f"Extracted {len(texts)} positive texts for {axis}.")
    return texts

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
    # Enforce mid-layer restriction (4-29)
    for L in priors:
        if not (4 <= L <= 29):
            priors[L] = 0.0
    max_w = max(priors.values())
    if max_w > 1e-8:
        for L in priors:
            priors[L] /= max_w
    return priors

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
    
    # 1. Load evaluation prompts
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
        
        # Normalize to unit vectors
        prompt_h_dict = {}
        for L in layer_indices:
            h_input = saved_h[L]
            h_unit = h_input / (np.linalg.norm(h_input) + 1e-10)
            prompt_h_dict[L] = h_unit
        prompts_h.append(prompt_h_dict)

    input_dir = Path("exp_steering_layer_analysis/results")
    v_data = np.load("vectors/mean_diff_vectors.npz")
    TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

    for axis in TRAITS:
        print(f"\n======================================================================")
        print(f"=== ANALYZING AXIS: {axis.upper()} ===")
        print(f"======================================================================")

        # 2. Load layer priors for this axis
        layer_priors = load_layer_priors(input_dir, axis)
        
        # 3. Load vectors and pre-saved activations
        layer_w = {}
        h_pos_dict = {}
        h_pos_all = {L: [] for L in layer_indices}
        for L in layer_indices:
            w_key = f"{L}|{axis}|w"
            if w_key in v_data:
                layer_w[L] = v_data[w_key]
            
            # Load pre-saved positive activations and mean
            H_pos_key = f"{L}|{axis}|H_pos_30"
            h_pos_key = f"{L}|{axis}|h_pos_30"
            if H_pos_key in v_data:
                H_pos = v_data[H_pos_key]
                h_pos_dict[L] = v_data[h_pos_key]
                h_pos_all[L].append(H_pos)

        cosine_selections = []
        cosine_prior_selections = []
        rank_selections = []
        rank_unconstrained_selections = []
        all_cos_similarities = {L: [] for L in layer_indices}
        
        # We will sweep over multiple penalty coefficients
        PENALTIES = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.0]
        
        # Structures to hold selections for each penalty
        cosine_prior_selections_by_penalty = {p: [] for p in PENALTIES}
        rank_selections_by_penalty = {p: [] for p in PENALTIES}
        
        cosine_selections = []
        rank_unconstrained_selections = []
        
        for p_idx, p_text in enumerate(prompts):
            h_dict = prompts_h[p_idx]
            
            cos_scores = {}
            percentiles = {}
            
            for L in layer_indices:
                h_unit = h_dict[L]
                
                # 1. Standard Cosine similarity (normalized)
                if L in layer_w:
                    w_vec = layer_w[L]
                    w_unit = w_vec / (np.linalg.norm(w_vec) + 1e-10)
                    cos_scores[L] = np.dot(h_unit, w_unit)
                else:
                    cos_scores[L] = -1.0
                
                # 2. Rank-based percentile similarity against Hh_pos
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
            
            # Unconstrained selections
            best_cos_layer = max(cos_scores, key=lambda L: cos_scores[L])
            cosine_selections.append(best_cos_layer)
            
            best_rank_uncon_layer = min(percentiles, key=lambda L: percentiles[L])
            rank_unconstrained_selections.append(best_rank_uncon_layer)
            
            # Save standard cos sims
            for L in layer_indices:
                all_cos_similarities[L].append(cos_scores[L])
                
            # Compute selections for each penalty coefficient
            for penalty in PENALTIES:
                cos_prior_scores = {}
                rank_scores = {}
                
                for L in layer_indices:
                    w_prior = layer_priors.get(L, 0.0)
                    
                    # Cosine-Prior
                    if w_prior > 1e-5:
                        cos_prior_scores[L] = cos_scores[L] - (1.0 - w_prior) * penalty
                    else:
                        cos_prior_scores[L] = -999.0
                        
                    # Rank-based
                    if L in h_pos_dict and w_prior > 1e-5:
                        rank_scores[L] = percentiles[L] + (1.0 - w_prior) * penalty
                    else:
                        rank_scores[L] = 999.0
                
                best_cos_prior = max(cos_prior_scores, key=lambda L: cos_prior_scores[L])
                best_rank = min(rank_scores, key=lambda L: rank_scores[L])
                
                cosine_prior_selections_by_penalty[penalty].append(best_cos_prior)
                rank_selections_by_penalty[penalty].append(best_rank)

        print(f"\n=== Layer-wise Cosine Similarity Calibration Stats for {axis.upper()} ===")
        print(f"{'Layer':<6} | {'Cos Mean':<10} | {'Cos Std':<8} | {'Prior Weight':<12}")
        print("-" * 46)
        for L in layer_indices:
            cos_mean = np.mean(all_cos_similarities[L]) if all_cos_similarities[L] else 0.0
            cos_std = np.std(all_cos_similarities[L]) if all_cos_similarities[L] else 0.0
            prior_w = layer_priors.get(L, 0.0)
            print(f"L{L:<4d} | {cos_mean:<10.4f} | {cos_std:<8.4f} | {prior_w:<12.4f}")

        # Helper function to compute Shannon entropy of a distribution
        def compute_entropy(selections):
            counts = pd.Series(selections).value_counts()
            probs = counts / len(selections)
            return -np.sum(probs * np.log2(probs))

        print(f"\n=== Selection Distribution Summary for {axis.upper()} ===")
        print(f"Standard Cosine Selected Layers (Unconstrained) (Entropy={compute_entropy(cosine_selections):.4f}):")
        print(f"  {cosine_selections}")
        print(f"Rank-based Selected Layers (Unconstrained) (Entropy={compute_entropy(rank_unconstrained_selections):.4f}):")
        print(f"  {rank_unconstrained_selections}")
        
        print("\n--- Sweeping Penalty Coefficients ---")
        print(f"{'Penalty':<8} | {'Cosine-Prior Selections':<45} | {'Entropy':<8} | {'Rank-based Selections':<45} | {'Entropy':<8}")
        print("-" * 125)
        for penalty in PENALTIES:
            cos_sel = cosine_prior_selections_by_penalty[penalty]
            cos_ent = compute_entropy(cos_sel)
            rank_sel = rank_selections_by_penalty[penalty]
            rank_ent = compute_entropy(rank_sel)
            
            cos_str = str(cos_sel)
            rank_str = str(rank_sel)
            print(f"{penalty:<8.1f} | {cos_str:<45} | {cos_ent:<8.4f} | {rank_str:<45} | {rank_ent:<8.4f}")

if __name__ == "__main__":
    main()