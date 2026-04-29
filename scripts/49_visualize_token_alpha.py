#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 49_visualize_token_alpha.py
#
# トークンごとに適用された Adaptive Steering の強度(alpha)と
# 境界からの距離(distance)を可視化する。
#

import argparse
import torch
import numpy as np
import yaml
import json
from pathlib import Path
from dataclasses import dataclass

from persona_vectors.live_axes import (
    load_model_and_tokenizer,
    _infer_main_device,
    get_layer_stack,
    _format_prompt
)

@dataclass
class VisualizingSteerer:
    model: torch.nn.Module
    layer: int
    w: torch.Tensor
    b: float
    tau: float
    max_alpha: float
    
    def __post_init__(self):
        self.handle = None
        self.history = [] # List of (token_id, alpha, dist)

    def register(self):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # 生成フェーズ（seq_len=1）のみを対象とする
            if hs.size(1) != 1:
                return out

            orig_dtype = hs.dtype
            hs_f32 = hs.to(torch.float32)
            device = hs.device
            w_dev = self.w.to(device)

            # d(h) = w \cdot h + b
            dot_product = (hs_f32 * w_dev).sum(dim=-1)
            dist = (dot_product + self.b).item()
            
            # alpha = clamp(tau - dist, 0, max_alpha)
            alpha_val = max(0.0, min(self.max_alpha, self.tau - dist))
            
            # 記録 (あとでトークンと紐付けるために保存)
            self.history.append({"alpha": alpha_val, "dist": dist})
            
            # ステアリング適用
            alpha_t = torch.tensor([alpha_val], device=device, dtype=torch.float32).view(1, 1)
            add = w_dev.view(1, 1, -1)
            steered = hs_f32 + alpha_t.unsqueeze(-1) * add
            
            steered = steered.to(orig_dtype)
            if isinstance(out, tuple):
                return (steered, *out[1:])
            return steered

        stack, _, _ = get_layer_stack(self.model)
        self.handle = stack[self.layer].register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--vector_bank", required=True)
    parser.add_argument("--prompt", default="Tell me about your day.")
    parser.add_argument("--axis", default="extraversion")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--max_alpha", type=float, default=5.0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading model: {cfg['model_name']}...")
    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], quant=cfg.get("quant", "auto"))
    model.eval()
    device = _infer_main_device(model)

    # 境界ベクトルのロード
    v_data = np.load(args.vector_bank)
    w = torch.tensor(v_data[f"{args.layer}|{args.axis}|w"], dtype=torch.float32)
    b = float(v_data[f"{args.layer}|{args.axis}|b"][0])

    steerer = VisualizingSteerer(model, args.layer, w, b, tau=args.tau, max_alpha=args.max_alpha)
    
    print(f"\nSteering Axis: {args.axis}, Layer: {args.layer}, Tau: {args.tau}")
    print(f"Prompt: {args.prompt}")
    
    formatted = _format_prompt(tokenizer, args.prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    # 生成
    steerer.register()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    steerer.remove()

    prompt_len = inputs.input_ids.shape[1]
    gen_ids = outputs[0][prompt_len:]
    
    print("\n=== Token-level Steering Analysis ===")
    print(f"{'Token':<15} | {'Alpha':<8} | {'Dist':<8} | {'Intensity'}")
    print("-" * 60)
    
    for i, token_id in enumerate(gen_ids):
        token_str = tokenizer.decode([token_id])
        token_display = token_str.replace('\n', '\\n')
        if i < len(steerer.history):
            h = steerer.history[i]
            alpha = h["alpha"]
            dist = h["dist"]
            # 可視化用のバー
            bar = "█" * int(alpha * 4)
            print(f"{token_display:<15} | {alpha:8.3f} | {dist:8.3f} | {bar}")
        else:
            print(f"{token_str:<15} | (N/A)")

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nFull Response:")
    print("-" * 40)
    print(full_text)

if __name__ == "__main__":
    main()
