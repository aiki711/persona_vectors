import torch
import numpy as np
import argparse
import os
import csv
import json
from contextlib import ExitStack
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from persona_vectors.live_axes import AnalysisSteerer, get_layer_stack, _ensure_pad_token

# Helper to get vector
def get_vector(data, layer, trait, device):
    key = f"{layer}|{trait}"
    if key in data:
        # Always use float32 for analysis precision and consistency
        v = torch.tensor(data[key], dtype=torch.float32, device=device)
        return v / (v.norm() + 1e-12)
    return None

def analyze_internal_states(args):
    print(f"=== Internal State Analysis (Online Metrics) for {args.model_id} ===")
    
    # 1. Load Model & Tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    tokenizer = _ensure_pad_token(tokenizer, model)

    # 2. Load Vector
    print(f"Loading vector from {args.vector_path}...")
    try:
        data = np.load(args.vector_path)
    except Exception as e:
        print(f"Error loading vector file: {e}")
        return

    # 3. Prepare Prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
         with open(args.prompt_file, 'r') as f:
             prompts = json.load(f)
         if args.limit > 0:
             prompts = prompts[:args.limit]
         print(f"Loaded {len(prompts)} prompts from file.")
    else:
        # Default simple prompts
        prompts = [
            "Hello, how are you?", 
            "What is the meaning of life?",
            "Tell me a joke.",
            "Explain quantum physics."
        ][:args.limit] if args.limit > 0 else [
            "Hello, how are you?", 
            "What is the meaning of life?",
            "Tell me a joke.",
            "Explain quantum physics."
        ]


    # Output setup
    fieldnames = [
        "PromptID", "Layer", "Trait", "Alpha", 
        "Sim_Input", "Sim_Proc", "Sim_Global", 
        "Norm_Input", "Norm_Proc", "Norm_Global",
        "Marginal_Sim_Global"
    ]
    
    # Check if output file exists to write header
    write_header = not os.path.exists(args.out_file)
    f_out = open(args.out_file, 'a', newline='') # Append mode
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # Parse layers
    stack, num_layers, _ = get_layer_stack(model)
    if "-" in args.steer_layers:
        s, e = map(int, args.steer_layers.split("-"))
        steer_target_layers = list(range(s, e+1))
    else:
        steer_target_layers = [int(x) for x in args.steer_layers.split(",")]
    
    print(f"Steering layers: {steer_target_layers}")
    
    # --- Processing Loop (Per Prompt) ---
    alphas = [float(x) for x in args.alpha.split(",")]
    print(f"Alphas: {alphas}")

    for pid, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        target_idx = -1 # Last token

        # --- A. Run Base Model & Cache ---
        base_cache = {} # Layer -> tensor(H) on GPU
        
        def make_base_hook(L, storage):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                # Capture last token, detach, keep on device, float32
                storage[L] = hs[0, target_idx].detach().to(dtype=torch.float32)
            return hook

        handles = []
        for i in range(num_layers):
            h = stack[i].register_forward_hook(make_base_hook(i, base_cache))
            handles.append(h)
        
        with torch.no_grad():
            model(**inputs)
        
        for h in handles: h.remove()
        
        # --- B. Run Steered Model (Per Alpha) ---
        for alpha in alphas:
            
            metrics_storage = [] 

            def make_analysis_callback(L, v_vec):
                temp_state = {} 
                
                def callback(event_type, tensor):
                    # tensor is (B, T, H), on device. Ensure float32.
                    vec = tensor[0, target_idx].detach().to(dtype=torch.float32)
                    
                    if event_type == 'in_pre':
                        temp_state['in_pre'] = vec
                    
                    elif event_type == 'in_post':
                        temp_state['in_post'] = vec
                        # Compute Input Delta: Post - Pre
                        if 'in_pre' in temp_state:
                            delta = vec - temp_state['in_pre']
                            if v_vec is not None:
                                sim = torch.dot(delta, v_vec).item()
                            else:
                                sim = 0.0
                            norm = delta.norm().item()
                            temp_state['sim_input'] = sim
                            temp_state['norm_input'] = norm
                            
                    elif event_type == 'out':
                        temp_state['out'] = vec
                        # Compute Proc Delta: Out - In_Post
                        if 'in_post' in temp_state:
                            delta = vec - temp_state['in_post']
                            if v_vec is not None:
                                sim = torch.dot(delta, v_vec).item()
                            else:
                                sim = 0.0
                            norm = delta.norm().item()
                            temp_state['sim_proc'] = sim
                            temp_state['norm_proc'] = norm
                        
                        # Compute Global Delta: Out - Base
                        if L in base_cache:
                            base_vec = base_cache[L]
                            delta = vec - base_vec
                            if v_vec is not None:
                                sim = torch.dot(delta, v_vec).item()
                            else:
                                sim = 0.0
                            norm = delta.norm().item()
                            temp_state['sim_global'] = sim
                            temp_state['norm_global'] = norm
                        
                        # Store final row data
                        metrics_storage.append({
                            "Layer": L,
                            "Sim_Input": temp_state.get('sim_input', 0.0),
                            "Sim_Proc": temp_state.get('sim_proc', 0.0),
                            "Sim_Global": temp_state.get('sim_global', 0.0),
                            "Norm_Input": temp_state.get('norm_input', 0.0),
                            "Norm_Proc": temp_state.get('norm_proc', 0.0),
                            "Norm_Global": temp_state.get('norm_global', 0.0),
                        })

                return callback

            # Register steerers
            steerers = []
            
            with ExitStack() as stack_ctx:
                # 1. Target Layers (Active Steering)
                for L in steer_target_layers:
                    key = f"{L}|{args.trait}"
                    v_np = data[key] if key in data else np.zeros(model.config.hidden_size)
                    v_vec = get_vector(data, L, args.trait, device)
                    
                    cb = make_analysis_callback(L, v_vec)
                    
                    st = AnalysisSteerer(
                        model=model,
                        layer=L,
                        v_mix=v_np,
                        alpha=alpha,
                        callback=cb,
                        answer_only=False 
                    )
                    stack_ctx.enter_context(st)
                
                # 2. Non-Target Layers (Passive Observation)
                all_layers = list(range(num_layers))
                for L in all_layers:
                    if L in steer_target_layers: continue
                    
                    v_vec = get_vector(data, L, args.trait, device)
                    cb = make_analysis_callback(L, v_vec)
                    
                    st = AnalysisSteerer(
                        model=model,
                        layer=L,
                        v_mix=np.zeros(model.config.hidden_size), 
                        alpha=0.0,
                        callback=cb,
                        answer_only=False
                    )
                    stack_ctx.enter_context(st)

                # Run
                with torch.no_grad():
                    model(**inputs)
            
            # Post-process metrics (Sort by layer, calc marginal)
            metrics_storage.sort(key=lambda x: x['Layer'])
            
            for i, m in enumerate(metrics_storage):
                if i == 0:
                    marginal = m['Sim_Global']
                else:
                    marginal = m['Sim_Global'] - metrics_storage[i-1]['Sim_Global']
                
                row = {
                    "PromptID": pid,
                    "Layer": m['Layer'],
                    "Trait": args.trait,
                    "Alpha": alpha,
                    "Sim_Input": m['Sim_Input'],
                    "Sim_Proc": m['Sim_Proc'],
                    "Sim_Global": m['Sim_Global'],
                    "Norm_Input": m['Norm_Input'],
                    "Norm_Proc": m['Norm_Proc'],
                    "Norm_Global": m['Norm_Global'],
                    "Marginal_Sim_Global": marginal
                }
                writer.writerow(row)
            
            f_out.flush()
            
    f_out.close()
    print(f"Analysis saved to {args.out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts")
    parser.add_argument("--alpha", type=str, default="5.0", help="Comma separated alphas")
    parser.add_argument("--steer_layers", type=str, default="10-20")
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--out_file", type=str, default="internal_states.csv")
    
    args = parser.parse_args()
    analyze_internal_states(args)
