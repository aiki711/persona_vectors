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
TEST_PROMPTS_LIMIT = 5
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
            if 10 <= L <= 22:
                priors[L] = 1.0
            else:
                priors[L] = 0.0
    # Enforce mid-layer restriction (10-22)
    for L in priors:
        if not (10 <= L <= 22):
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
    
    # 1. Load layer priors (Prior mask)
    input_dir = Path("exp_steering_layer_analysis/results")
    print(f"Loading layer priors from {input_dir}...")
    layer_priors = load_layer_priors(input_dir, AXIS)
    
    # 2. Extract positive texts
    pos_texts = extract_positive_texts(AXIS, limit=N_SAMPLES)
    
    # 3. Extract hidden states for positive texts to compute h_pos
    print("Extracting positive hidden states...")
    h_pos_all = {L: [] for L in layer_indices}
    
    @torch.no_grad()
    def get_hidden_states(texts):
        msgs_prefix = [{"role": "user", "content": "Hello."}]
        prefix_ids = tok.apply_chat_template(msgs_prefix, add_generation_prompt=True, tokenize=True)
        len_prefix = len(prefix_ids)
        
        full_inputs = []
        for t in texts:
            msgs = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": t}]
            full_ids = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
            full_inputs.append(torch.tensor(full_ids))
            
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(full_inputs, batch_first=True, padding_value=tok.pad_token_id).to(device)
        attn_mask = (input_ids != tok.pad_token_id).long()
        out = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        
        results = {L: [] for L in layer_indices}
        for b in range(len(texts)):
            s_idx, e_idx = len_prefix, attn_mask[b].sum().item()
            if s_idx >= e_idx: s_idx = e_idx - 1
            for L in layer_indices:
                results[L].append(out.hidden_states[L][b][s_idx:e_idx].mean(dim=0))
        return {L: torch.stack(v) for L, v in results.items()}

    batch_size = 5
    for i in range(0, len(pos_texts), batch_size):
        batch = pos_texts[i:i+batch_size]
        out_hs = get_hidden_states(batch)
        for L in layer_indices:
            h_pos_all[L].append(out_hs[L].cpu().numpy())
            
    h_pos_dict = {}
    for L in layer_indices:
        H_pos = np.concatenate(h_pos_all[L], axis=0) # [30, n_dims]
        h_pos = np.mean(H_pos, axis=0, keepdims=True) # [1, n_dims]
        h_pos_dict[L] = h_pos

    # 4. Load test prompts and pre-compute their hidden states to build the reference distribution
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
    
    print(f"Extracting hidden states for {len(all_prompts)} test prompts for distribution calibration...")
    h_test_all = {L: [] for L in layer_indices}
    for p_text in all_prompts:
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
            for h in handles: h.remove()
        for L in layer_indices:
            h_test_all[L].append(saved_h[L])
            
    sims_test_dict = {}
    for L in layer_indices:
        H_test = np.stack(h_test_all[L], axis=0) # [N_PROMPTS, n_dims]
        H_test_norm = H_test / (np.linalg.norm(H_test, axis=1, keepdims=True) + 1e-10)
        h_norm = h_pos_dict[L] / (np.linalg.norm(h_pos_dict[L]) + 1e-10)
        sims_test_dict[L] = np.dot(H_test_norm, h_norm.T).squeeze() # [N_PROMPTS]

    prompts = all_prompts[:TEST_PROMPTS_LIMIT]
    
    # Load vectors (for standard Cosine Sim comparison)
    v_data = np.load("vectors/mean_diff_vectors.npz")
    layer_w = {}
    for L in layer_indices:
        w_key = f"{L}|{AXIS}|w"
        if w_key in v_data:
            layer_w[L] = v_data[w_key]

    print("\nRunning layer selection comparison on test prompts...")
    
    cosine_selections = []
    rank_selections = []
    
    for p_idx, p_text in enumerate(prompts):
        print(f"\n--- Prompt {p_idx+1}: {p_text[:60]}... ---")
        
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
                
            cos_scores = {}
            rank_scores = {}
        
        for L in layer_indices:
            h_input = saved_h[L]
            h_unit = h_input / (np.linalg.norm(h_input) + 1e-10)
            
            # 1. Cosine similarity
            if L in layer_w:
                w_vec = layer_w[L]
                cos_scores[L] = np.dot(h_unit, w_vec)
            else:
                cos_scores[L] = -1.0
                
            # 2. Normalized Rank Score (Input vs Calibration distribution)
            h_pos_norm = h_pos_dict[L] / (np.linalg.norm(h_pos_dict[L]) + 1e-10)
            sim_input = np.dot(h_unit, h_pos_norm.T).item()
            
            sims_pos = sims_test_dict[L] # [N_PROMPTS]
            sims_combined = np.concatenate([sims_pos, [sim_input]]) # [N_PROMPTS+1]
            ranking = np.argsort(sims_combined)
            
            # Index where input similarity lands (0 = lowest, N_PROMPTS = highest)
            rank_idx = np.where(ranking == len(sims_combined) - 1)[0][0]
            percentile = rank_idx / float(len(sims_pos))
            
            w_prior = layer_priors.get(L, 0.0)
            if w_prior > 1e-5:
                rank_scores[L] = percentile + (1.0 - w_prior) * 10.0
            else:
                rank_scores[L] = 999.0 # Completely exclude
            
        # Select best layers
        best_cos_layer = max(cos_scores, key=lambda L: cos_scores[L])
        best_rank_layer = min(rank_scores, key=lambda L: rank_scores[L])
        
        cosine_selections.append(best_cos_layer)
        rank_selections.append(best_rank_layer)
        
        print(f"  Standard Cosine Selection : Layer {best_cos_layer:2d} (Score={cos_scores[best_cos_layer]:.4f})")
        print(f"  Rank-based Selection (Min): Layer {best_rank_layer:2d} (Percentile={rank_scores[best_rank_layer]:.4f})")
        
        # Print top 5 layers (sorted)
        top_cos = sorted(cos_scores.keys(), key=lambda L: cos_scores[L], reverse=True)[:5]
        top_rank = sorted(rank_scores.keys(), key=lambda L: rank_scores[L], reverse=False)[:5]
        
        print("  Top 5 Cosine Layers       : " + ", ".join(f"L{L}({cos_scores[L]:.3f})" for L in top_cos))
        print("  Top 5 Rank Layers (Min)   : " + ", ".join(f"L{L}(rank={rank_scores[L]:.3f}, pct={rank_scores[L] - (1.0-layer_priors.get(L,0.0))*10.0:.3f})" for L in top_rank if rank_scores[L] < 5.0))
        
    print("\n=== Selection Distribution Summary ===")
    print("Standard Cosine Selected Layers: ", cosine_selections)
    print("Rank-based Selected Layers (Min):", rank_selections)

if __name__ == "__main__":
    main()