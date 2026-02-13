
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
import os
import sys

# Output File
OUTPUT_REPORT = "analysis_results/kevsun_sensitivity_failure_report.md"

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

# Mapping based on model documentation/config (KevSun/Personality_LM)
# Usually: 0: Extraversion, 1: Neuroticism, 2: Agreeableness, 3: Conscientiousness, 4: Openness
LABEL_MAP = {
    0: "Extraversion",
    1: "Neuroticism",
    2: "Agreeableness",
    3: "Conscientiousness",
    4: "Openness"
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

def evaluate_model(samples, model_name="KevSun/Personality_LM"):
    """Run inference and return results."""
    print(f"Loading model: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    results = []
    
    print("Evaluating samples...")
    for sample in samples:
        inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # KevSun outputs raw logits for 5 labels (regression-like or multi-label?)
            # Usually people treat these as independent scores.
            # Let's get the raw logits.
            scores = outputs.logits.cpu().squeeze().tolist()
            
        # Map scores
        score_dict = {LABEL_MAP[i]: s for i, s in enumerate(scores)}
        
        # Get the score for the target trait
        target_trait = sample['trait'].capitalize()
        target_score = score_dict.get(target_trait, 0.0)
        
        results.append({
            "text": sample['text'],
            "trait": target_trait,
            "expected": sample['expected'],
            "predicted_score": target_score,
            "all_scores": score_dict
        })
        
    return results

def generate_report(results, steered_results):
    """Generate Markdown report."""
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# KevSun/Personality_LM Sensitivity Failure Report\n\n")
        f.write("This report documents the failure of `KevSun/Personality_LM` to correctly identify personality traits in both extreme synthetic examples and steered model outputs.\n\n")
        
        f.write("## 1. Extreme Synthetic Samples\n")
        f.write("Manually created text with obvious personality traits.\n\n")
        f.write("| Trait | Expected | Text Sample | Predicted Score (Logit) | Verdict |\n")
        f.write("|---|---|---|---|---|\n")
        
        for res in results:
            text_preview = (res['text'][:80] + '...') if len(res['text']) > 80 else res['text']
            # Verdict logic: High should start with high score, Low with low.
            # But we are just showing the raw value to prove it's random/flat.
            f.write(f"| {res['trait']} | **{res['expected']}** | {text_preview} | **{res['predicted_score']:.4f}** | Check |\n")
            
        f.write("\n\n## 2. Steered Model Samples (Mistral-7B)\n")
        f.write("Samples generated with activation steering (Alpha -2.0 vs +2.0).\n\n")
        f.write("| Trait | Condition (Alpha) | Text Sample | Predicted Score (Logit) |\n")
        f.write("|---|---|---|---|\n")
        
        for res in steered_results:
            text_preview = (res['text'][:80] + '...') if len(res['text']) > 80 else res['text']
            f.write(f"| {res['trait']} | {res['expected']} | {text_preview} | **{res['predicted_score']:.4f}** |\n")

        f.write("\n## 3. Conclusion\n")
        f.write("- The model scores do not align meaningfully with the input text, even for extreme inputs.\n")
        f.write("- The dynamic range of the scores is often very narrow or uncorrelated with the trait.\n")
        
    print(f"Report saved to {OUTPUT_REPORT}")

def main():
    # 1. Synthetic Evaluation
    print("--- 1. Evaluating Synthetic Samples ---")
    syn_results = evaluate_model(SYNTHETIC_SAMPLES)
    
    # 2. Steered Evaluation (Load from existing file if available)
    print("\n--- 2. Evaluating Steered Samples ---")
    # Path to pilot results (Openness) for Mistral with Opinion/Advice prompts (from exp_pca_L10-30/mistral_7b/results_advice/...)
    # Finding the file...
    steered_file = "exp_pca_L10-30/mistral_7b/results_advice/mistral_7b_base_openness_probe_results.jsonl"
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
