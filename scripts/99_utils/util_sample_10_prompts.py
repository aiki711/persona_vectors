#!/usr/bin/env python3
import json
import random
import os

random.seed(42)

in_file = "exp_adaptive_steering/results/test_prompts.jsonl"
out_file = "exp_adaptive_steering/results/test_prompts_10.jsonl"

print(f"Sampling 10 prompts from {in_file}...")
with open(in_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

sampled = random.sample(lines, 10)

with open(out_file, 'w', encoding='utf-8') as f:
    for line in sampled:
        f.write(line)

print(f"Saved {len(sampled)} prompts to {out_file}.")
