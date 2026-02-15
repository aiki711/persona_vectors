# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from persona_vectors.live_axes import load_model_and_tokenizer, build_axes_for_model, generate_with_steer, AXES

# 設定
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LAYERS = list(range(15, 26)) # 15 to 25
THETAS = [-0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8] # 7 steps, symmetric
TRAITS = AXES
PROMPTS = [
    "Tell me about your typical weekend.",
    "What do you think about trying new things and exploring the unknown?",
    "How do you handle a situation where someone disagrees with you?",
]

OUTPUT_ROOT = Path("exp_angular_steering") / "mistral-7b-v0.2"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def main():
    print("="*50)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Theta Range: {THETAS}")
    print("="*50)
    
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, quant="4bit")
    
    # 軸ベクトルの準備
    pos_texts = {ax: ["I am very " + ax] for ax in AXES}
    neg_texts = {ax: ["I am not " + ax] for ax in AXES}
    
    print("Building axes for selected layers...")
    axes_all_layers = build_axes_for_model(
        model, tokenizer, layer_list=LAYERS, 
        pos_texts=pos_texts, neg_texts=neg_texts
    )

    for trait in TRAITS:
        print(f"Steering for trait: {trait}")
        e = {trait: 1.0}
        
        output_file = OUTPUT_ROOT / f"{trait}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(PROMPTS):
                for layer in LAYERS:
                    for theta in THETAS:
                        output = generate_with_steer(
                            model, tokenizer, prompt, axes_all_layers, layer, 
                            theta=theta, e=e, mode="angular", max_new_tokens=128
                        )
                        res = {
                            "i": i,
                            "trait": trait,
                            "layers": [layer],
                            "alpha_total": theta,
                            "alpha_mode": "angular",
                            "alpha_per_layer": theta,
                            "x": prompt,
                            "y": output,
                            "s_avg": None,
                            "s0_avg": None,
                            "ds_avg": None,
                            "s_by_layer": {}
                        }
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        f.flush()

if __name__ == "__main__":
    main()
