
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import os
import sys
import re
from tqdm import tqdm

# Output File
OUTPUT_REPORT = "analysis_results/llama3_sensitivity_report.md"

# Extreme Synthetic Samples (Clearly High/Low traits)
SYNTHETIC_SAMPLES = [
    {
        "text": "I absolutely love going to huge parties! The more people, the better. I can talk to strangers all night long.",
        "trait": "Extraversion",
        "expected": "High"
    },
    {
        "text": "Please leave me alone. I hate social interaction and prefer to stay in my room in complete silence.",
        "trait": "Extraversion",
        "expected": "Low"
    },
    {
        "text": "I have so many new ideas for inventions! I love exploring abstract concepts and trying exotic foods.",
        "trait": "Openness",
        "expected": "High"
    },
    {
        "text": "I only like doing things the traditional way. I hate change and abstract art makes no sense to me.",
        "trait": "Openness",
        "expected": "Low"
    },
    {
        "text": "I am extremely organized. I have a schedule for every minute of my day and my desk is perfectly clean.",
        "trait": "Conscientiousness",
        "expected": "High"
    },
     {
        "text": "I completely forgot about the deadline. My room is a mess and I can't find anything.",
        "trait": "Conscientiousness",
        "expected": "Low"
    }
]

# Big Five Definitions for Prompting (Same as in 14_calc_personality_score_llm.py)
TRAIT_DEFINITIONS = {
    "extraversion": "Extraversion reflects an individual's sociability, assertiveness, and enthusiasm. High scorers are outgoing and energetic; low scorers are solitary and reserved.",
    "neuroticism": "Neuroticism reflects emotional instability and tendency to experience negative emotions. High scorers are anxious and moody; low scorers are calm and confident.",
    "agreeableness": "Agreeableness reflects an individual's tendency to be compassionate and cooperative. High scorers are trusting and helpful; low scorers are competitive and critical.",
    "conscientiousness": "Conscientiousness reflects an individual's level of self-discipline and organization. High scorers are efficient and organized; low scorers are extravagant and careless.",
    "openness": "Openness reflects an individual's intellectual curiosity and creative imagination. High scorers are inventive and curious; low scorers are consistent and cautious."
}

def load_steered_samples(jsonl_path, trait, num_samples=3):
    """Load samples from actual experiment results for comparison."""
    if not os.path.exists(jsonl_path):
        print(f"Warning: File not found {jsonl_path}")
        return []
    
    samples = []
    try:
        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        # Get Alpha -2.0 and +2.0
        low_df = df[df['alpha_total'] == -2.0]
        high_df = df[df['alpha_total'] == 2.0]
        
        for _, row in low_df.head(num_samples).iterrows():
            samples.append({
                "text": row['y'],
                "trait": trait,
                "expected": "Calculated Low (Steered alpha=-2.0)",
                "source": "Steered Model"
            })
            
        for _, row in high_df.head(num_samples).iterrows():
            samples.append({
                "text": row['y'],
                "trait": trait,
                "expected": "Calculated High (Steered alpha=+2.0)",
                "source": "Steered Model"
            })
            
    except Exception as e:
        print(f"Error reading jsonl: {e}")
        
    return samples

def evaluate_model(samples, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Run inference and return results."""
    print(f"Loading Judge Model: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    results = []
    
    print("Evaluating samples...")
    for sample in tqdm(samples):
        trait = sample['trait']
        text = sample['text']
        
        trait_lower = trait.lower()
        definition = TRAIT_DEFINITIONS.get(trait_lower, "No definition available.")
        
        # System + User message structure for Llama-3
        system_msg = f"""You are an expert psychologist evaluating personality traits from text.

Trai: **{trait.capitalize()}**
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
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Parse Integer
        match = re.search(r'\b([1-5])\b', generated_text)
        if match:
            score_val = int(match.group(1))
        else:
            score_val = 3 # Neutral fallback

        results.append({
            "text": text,
            "trait": trait,
            "expected": sample['expected'],
            "predicted_score": score_val,
            "raw_output": generated_text
        })
        
    return results

def generate_report(results, steered_results):
    """Generate Markdown report."""
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# Llama-3 (LLM-as-a-Judge) Sensitivity Report\n\n")
        f.write("This report documents the sensitivity of `Meta-Llama-3-8B-Instruct` as a judge for personality traits.\n\n")
        
        f.write("## 1. Extreme Synthetic Samples\n")
        f.write("Manually created text with obvious personality traits.\n\n")
        f.write("| Trait | Expected | Text Sample | Predicted Score (1-5) | Verdict |\n")
        f.write("|---|---|---|---|---|\n")
        
        for res in results:
            text_preview = (res['text'][:80] + '...') if len(res['text']) > 80 else res['text']
            score = res['predicted_score']
            
            # Simple verdict check
            verdict = "OK"
            if res['expected'] == "High" and score >= 4: verdict = "✅ Pass"
            elif res['expected'] == "Low" and score <= 2: verdict = "✅ Pass"
            else: verdict = "⚠️ Questionable"
            
            f.write(f"| {res['trait']} | **{res['expected']}** | {text_preview} | **{score}** | {verdict} |\n")
            
        f.write("\n\n## 2. Steered Model Samples (Mistral-7B)\n")
        f.write("Samples generated with activation steering (Alpha -2.0 vs +2.0).\n\n")
        f.write("| Trait | Condition (Alpha) | Text Sample | Predicted Score (1-5) |\n")
        f.write("|---|---|---|---|\n")
        
        for res in steered_results:
            text_preview = (res['text'][:80] + '...') if len(res['text']) > 80 else res['text']
            # Highlight extreme scores
            score_str = f"**{res['predicted_score']}**"
            f.write(f"| {res['trait']} | {res['expected']} | {text_preview} | {score_str} |\n")

        f.write("\n## 3. Conclusion\n")
        f.write("- Unlike KevSun/Personality_LM, Llama-3 shows [strong/weak] correlation with the input text.\n")
        f.write("- Extreme synthetic samples are [correctly/incorrectly] classified.\n")
        
    print(f"Report saved to {OUTPUT_REPORT}")

def main():
    # 1. Synthetic Evaluation
    print("--- 1. Evaluating Synthetic Samples ---")
    syn_results = evaluate_model(SYNTHETIC_SAMPLES)
    
    # 2. Steered Evaluation (Load from existing file if available)
    print("\n--- 2. Evaluating Steered Samples ---")
    # Using the NEW personality expression results if available, else old advice
    # Try updated file first: exp_personality_L10-30/mistral_7b/results_personality_expression/mistral_7b_base_openness_probe_results.jsonl
    steered_file = "exp_personality_L10-30/mistral_7b/results_personality_expression/mistral_7b_base_openness_probe_results.jsonl"
    if not os.path.exists(steered_file):
         steered_file = "exp_pca_L10-30/mistral_7b/results_advice/mistral_7b_base_openness_probe_results.jsonl"
         
    print(f"Loading samples from: {steered_file}")
    
    steered_samples = load_steered_samples(steered_file, "Openness", num_samples=3)
    
    steered_results = []
    if steered_samples:
        steered_results = evaluate_model(steered_samples)
    else:
        print("Skipping steered samples (file not found or empty).")
        
    # 3. Report
    generate_report(syn_results, steered_results)

if __name__ == "__main__":
    main()
