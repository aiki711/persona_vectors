
import json
import os
import random
from datasets import load_dataset

def create_synthetic_prompts(out_path):
    prompts = [
        "Explain the process of photosynthesis in simple terms.",
        "What are the main differences between Python and Java?",
        "Write a short story about a robot who wants to become a painter.",
        "How do you make a perfect cup of coffee?",
        "Describe the plot of the movie 'The Matrix'.",
        "What is the capital of Australia?",
        "List three benefits of regular exercise.",
        "Translate 'Hello, how are you?' into French.",
        "Write a python function to check if a number is prime.",
        "What are the primary colors?",
        "Explain the theory of relativity to a 10-year-old.",
        "Give me a recipe for chocolate chip cookies.",
        "What is the significance of the Rosetta Stone?",
        "Draft a professional email to reschedule a meeting.",
        "How does a car engine work?",
        "Compare and contrast renewable and non-renewable energy sources.",
        "Who wrote 'Pride and Prejudice'?",
        "What are the symptoms of the common cold?",
        "Write a haiku about winter.",
        "Explain the concept of supply and demand.",
        "What is the distance between the Earth and the Moon?",
        "How do airplanes stay in the air?",
        "Summarize the main events of World War II.",
        "What is the difference between a virus and a bacterium?",
        "Write a poem about the ocean.",
        "How does the internet work?",
        "What are the 50 states of the USA?",
        "Explain the rules of chess.",
        "What is the function of the mitochondria?",
        "Who painted the Mona Lisa?",
        "Describe the water cycle.",
        "What is the tallest mountain in the world?",
        "How do I change a tire on a car?",
        "What is the freezing point of water in Fahrenheit?",
        "Write a brief history of the internet.",
        "What are the ingredients in a Caesar salad?",
        "How does a camera work?",
        "What is the definition of 'artificial intelligence'?",
        "List the planets in our solar system.",
        "What is the currency of Japan?",
        "How do trees communicate?",
        "Write a joke about a computer.",
        "What is the most spoken language in the world?",
        "Explain the concept of blockchain.",
        "What is the difference between weather and climate?",
        "How do I tie a tie?",
        "What is the meaning of life? (Philosophical)",
        "Write a short review of a fictional book.",
        "How do I bake bread?",
        "What are the benefits of meditation?"
    ]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(prompts)} synthetic prompts to {out_path}")

def create_mtbench_prompts(out_path):
    print("Loading MT-Bench dataset...")
    try:
        # Load mt_bench from HuggingFace (lmsys/mt_bench_human_judgments is one source, but usually prompts are in lmsys/mt_bench)
        # Using 'lmsys/mt_bench_human_judgments' as it is a common subset reference or similar. 
        # Actually 'HuggingFaceH4/mt_bench_prompts' is a good clean source for prompts.
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        
        # Select 50 prompts suitable for single-turn or reasoning. 
        # MT-Bench has categories. We want diversity.
        categories = dataset.unique("category")
        selected_prompts = []
        
        # Try to get balanced number from each category
        per_cat = 50 // len(categories) + 1
        
        for cat in categories:
            cat_data = dataset.filter(lambda x: x["category"] == cat)
            prompts = cat_data["prompt"] # list of prompts (some might be multi-turn, taking first turn)
            
            # Take first turn only if it is a list, otherwise take string
            cleaned_prompts = []
            for p in prompts:
                if isinstance(p, list):
                    cleaned_prompts.append(p[0])
                elif isinstance(p, str):
                    cleaned_prompts.append(p)
            
            # Subsample
            sample = cleaned_prompts[:per_cat]
            selected_prompts.extend(sample)
            
        # Cut to exactly 50
        random.shuffle(selected_prompts)
        final_prompts = selected_prompts[:50]
        
    except Exception as e:
        print(f"Error loading MT-Bench: {e}")
        print("Falling back to manual subset...")
        # Fallback list if HF fails/timeout
        final_prompts = [
            "Compose an engaging travel blog post about a recent trip to Hawaii.",
            "Write a letter to your future self 10 years from now.",
            "Tell me a joke about engineers.",
            "How does a rainbow form?",
            "What are the main arguments for and against universal basic income?",
            "Describe a vivid dream you had recently.",
            "Explain how a neural network learns.",
            "Write a python script to scrape a website.",
            "Design a workout plan for a beginner.",
            "What are the health benefits of green tea?",
            # ... (truncated for brevity in fallback) ...
        ]
        # Just filling up to 50 with duplicates for fallback test if needed, but error is printed.
        while len(final_prompts) < 50:
            final_prompts.append("Tell me something interesting.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(final_prompts)} MT-Bench prompts to {out_path}")

def main():
    create_synthetic_prompts("exp/01_probe_inputs/synthetic_50.json")
    create_mtbench_prompts("exp/01_probe_inputs/mtbench_50.json")

if __name__ == "__main__":
    main()
