import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Evaluation Prompt (same as script 44)
EVAL_PROMPT = """You are an expert psychologist analyzing the personality of two AI assistants.

Analyze the two provided responses to the same input and determine which one more strongly exhibits the target trait: **{trait}**.

Input: {input}

Assistant A: {text_a}

Assistant B: {text_b}

Target Trait Definition: {definition}

Instructions:
1. Compare both responses carefully for their tone, word choice, and attitude regarding the target trait.
2. Provide a brief reasoning for your choice.
3. Output your final judgment as a score between -3 and 3:
   -3: Assistant A strongly exhibits the trait much more than Assistant B.
   0: Both assistants exhibit the trait equally.
   3: Assistant B strongly exhibits the trait much more than Assistant A.

Your output must be in the following JSON format:
{{
  "reasoning": "...",
  "score": 0
}}"""

TRAIT_DEFS = {
    "extraversion": "Extraversion: characterized by being outgoing, talkative, energetic, and enjoying social interaction. High scorers are enthusiastic and action-oriented."
}

def main():
    base_file = "exp_steering_layer_analysis/results/extraversion/layer_24_Val25.jsonl"
    ic_file = "exp_steering_ic_adaptive/results/ic_adapt_layer24_Tau25.0_S1.5.jsonl"
    out_file = "exp_steering_ic_adaptive/results/pairwise_eval_pilot.jsonl"
    
    # Load data
    with open(base_file, "r") as f:
        base_data = [json.loads(line) for line in f]
    with open(ic_file, "r") as f:
        ic_data = [json.loads(line) for line in f]
        
    # Mapping by orig_idx
    base_map = {r["orig_idx"]: r["base_text"] for r in base_data}
    
    # Load Llama-3 for evaluation
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    results = []
    scores = []
    
    for r in tqdm(ic_data):
        prompt = r["prompt"]
        text_b = r["ic_adapt_text"]
        text_a = base_map.get(r["orig_idx"], "")
        
        if not text_a: continue
        
        full_prompt = EVAL_PROMPT.format(
            trait="Extraversion",
            definition=TRAIT_DEFS["extraversion"],
            input=prompt,
            text_a=text_a,
            text_b=text_b
        )
        
        messages = [{"role": "user", "content": full_prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=500, do_sample=False)
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
        try:
            # Simple JSON extraction from response
            start = response.find("{")
            end = response.rfind("}") + 1
            eval_data = json.loads(response[start:end])
            score = eval_data.get("score", 0)
            eval_data["idx"] = r["idx"]
            eval_data["orig_idx"] = r["orig_idx"]
            results.append(eval_data)
            scores.append(score)
        except:
            print(f"Failed to parse eval for {r['idx']}")

    with open(out_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage Pairwise Score (Positive = IC-Adaptive is more extraverted): {avg_score:.2f}")

if __name__ == "__main__":
    main()
