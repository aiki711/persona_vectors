import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
import os
import re

# Big Five Definitions for Prompting
TRAIT_DEFINITIONS = {
    "extraversion": "Extraversion reflects an individual's sociability, assertiveness, and enthusiasm. High scorers are outgoing and energetic; low scorers are solitary and reserved.",
    "neuroticism": "Neuroticism reflects emotional instability and tendency to experience negative emotions. High scorers are anxious and moody; low scorers are calm and confident.",
    "agreeableness": "Agreeableness reflects an individual's tendency to be compassionate and cooperative. High scorers are trusting and helpful; low scorers are competitive and critical.",
    "conscientiousness": "Conscientiousness reflects an individual's level of self-discipline and organization. High scorers are efficient and organized; low scorers are extravagant and careless.",
    "openness": "Openness reflects an individual's intellectual curiosity and creative imagination. High scorers are inventive and curious; low scorers are consistent and cautious."
}

def load_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    return pd.DataFrame(data)

def get_judge_score(model, tokenizer, texts, traits, device, batch_size=8):
    scores = []
    
    # Check lengths
    if len(texts) != len(traits):
        if isinstance(traits, str):
            traits = [traits] * len(texts)
        else:
            raise ValueError("Length of texts and traits must match.")

    print(f"Evaluating {len(texts)} samples with batch size {batch_size}...")
    
    # Prepare all prompts first
    prompts = []
    for text, trait in zip(texts, traits):
        trait_lower = trait.lower()
        definition = TRAIT_DEFINITIONS.get(trait_lower, "No definition available.")
        
        # System + User message structure for Llama-3
        system_msg = f"""You are an expert psychologist evaluating personality traits from text.

Trait: **{trait.capitalize()}**
Definition: {definition}

Evaluate the level of {trait.capitalize()} expressed in the text content.
Focus on the attitudes, opinions, behaviors, and emotional tone expressed, NOT on the persona or role being played.

Respond ONLY with a single integer from 1 to 5.

Scale:
1: Very Low {trait.capitalize()} (strong opposite traits)
2: Low {trait.capitalize()}
3: Neutral / Mixed
4: High {trait.capitalize()}
5: Very High {trait.capitalize()} (strong trait expression)"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Text: \"{text}\"\n\nScore:"}
        ]
        
        # Apply prompt template
        # output: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    # Batch Inference
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_traits = traits[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode only the new tokens
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            # Llama-3 padding is usually left-padded if configured, but let's be safe
            # Actually AutoTokenizer padding default varies. 
            # We decoded using skip_special_tokens=True for the WHOLE valid sequence
            # But simpler: just decode the newly generated part.
            
            # Note: outputs[j] includes input_ids. We need to slice.
            # But since padding affects indices, we must be careful.
            # 'inputs' is batched. 'outputs' is batched.
            # Safe way: decode everything and split by prompt end? Or slice by length?
            # Slicing by input_ids.shape[1] works if left-padded?
            # LlamaTokenizer usually rights pads by default unless set.
            # Let's decode the *whole* thing and regex search the last part.
            
            full_text = tokenizer.decode(output, skip_special_tokens=True)
            # The prompt is also decoded.
            # Only look at the generation. 
            # Llama-3 output format: ... Score: 5
            
            # Simple parsing: Find the LAST integer in the text?
            # Or better: slice off the prompt length?
            # Prompt length in tokens varies.
            
            # Let's try slicing input length. 
            # Wait, batch padding makes input_ids all same length. 
            # So outputs[:, input_ids.shape[1]:] is safe if we right-padded.
            
            generated_text = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Parse Integer
            match = re.search(r'\b([1-5])\b', generated_text)
            if match:
                score_val = int(match.group(1))
            else:
                # Fallback: check if the text contains a number word or just fails
                # print(f"Warning: Could not parse score from '{generated_text}'.")
                score_val = 3 # Neutral fallback
            
            trait_lower = batch_traits[j].lower()
            scores.append({
                f"score_{trait_lower}": (score_val - 1) / 4.0,
                f"raw_score_{trait_lower}": score_val
            })

    return pd.DataFrame(scores)

def main():
    parser = argparse.ArgumentParser(description="Evaluate personality scores using LLM-as-a-Judge.")
    parser.add_argument("input_file", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, default="personality_scores_llm.csv")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading Judge Model: {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Set padding token (Llama-3 doesn't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load Data
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    if 'y' not in df.columns:
        print("Error: 'y' column (text) missing.")
        sys.exit(1)
        
    # Determine trait
    # If output file has a specific trait in name, use that? 
    # Or rely on 'trait' column in jsonl if exists
    if 'trait' in df.columns:
        traits = df['trait'].tolist()
    else:
        # Fallback or error? 
        # For now, if missing, check filename or arg?
        # Let's assume input filename has trait?
        # Better: require 'trait' column which our probe results have.
        print("Error: 'trait' column missing in input JSONL. Cannot determine evaluation target.")
        sys.exit(1)

    # Evaluate
    texts = df['y'].fillna("").astype(str).tolist()
    scores_df = get_judge_score(model, tokenizer, texts, traits, device)

    # Merge
    print("Merging results...")
    result_df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    # Save
    print(f"Saving to {args.output}...")
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print("Done.")

if __name__ == "__main__":
    main()
