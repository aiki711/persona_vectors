
import json
import os
import random
from datasets import load_dataset

def create_writing_prompts(out_path):
    """
    Creates a dataset of 30 short creative writing prompts.
    Inspired by /r/WritingPrompts but curated for length and variety.
    """
    prompts = [
        "Write a diary entry from the perspective of a lighthouse keeper who hasn't seen a ship in years.",
        "You discover a door in your house that wasn't there yesterday. Describe what happens when you open it.",
        "Write a monologue for a villain explaining why they are actually the hero.",
        "Describe a color to someone who has been blind since birth.",
        "You are an astronaut on the first mission to Mars. Write your final log entry before returning home.",
        "Write a story starting with the sentence: 'The clock struck thirteen.'",
        "A time traveler visits you and gives you one piece of advice. What is it and how do you react?",
        "Describe your perfect day from morning to night.",
        "Write a letter to your childhood self.",
        "You find a wallet on the street with a large sum of money and a mysterious note. What do you do?",
        "Write a scene where two characters are breaking up but neither wants to say the words.",
        "Describe a bustling city market from the perspective of a stray cat.",
        "You wake up one morning with the ability to read minds, but you can't turn it off. Describe your commute to work.",
        "Write a review for a restaurant that serves emotions instead of food.",
        "You act as a historian in the year 3000 describing the 'primitive' technology of the smartphone.",
        "Write a diary entry about a day where everything went wrong, but ended perfectly.",
        "Describe the smell of rain after a long drought.",
        "You are a ghost haunting your old house. Describe your attempts to scare the new tenants.",
        "Write a conversation between the Sun and the Moon.",
        "You plant a mysterious seed you found in the attic. Describe what grows.",
        "Write a story about a world where music is illegal.",
        "You receive a text message from a number that died 5 years ago.",
        "Describe a room that reflects the personality of its owner without mentioning the owner.",
        "Write a speech for a politician who can only tell the truth.",
        "You are a character in a book who realizes they are being written. What do you do?",
        "Write a description of a forest made entirely of crystal.",
        "You gain the power to stop time, but only for 10 seconds at a time.",
        "Write a diary entry from the perspective of an ant.",
        "Describe the taste of your favorite memory.",
        "You are the last person on Earth. The phone rings."
    ]
    
    # Ensure exactly 30 prompts
    if len(prompts) != 30:
        print(f"Warning: Expected 30 writing prompts, but got {len(prompts)}.")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(prompts)} writing prompts to {out_path}")

def create_opinion_advice_prompts(out_path):
    """
    Creates a dataset of 30 opinion and advice prompts from MT-Bench.
    Filters categories 'writing', 'reasoning', 'roleplay' and selects relevant ones.
    """
    print("Loading MT-Bench dataset (from HuggingFace)...")
    try:
        # Using HuggingFaceH4/mt_bench_prompts as source
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        
        # Categories suitable for Opinion & Advice
        target_categories = ['writing', 'reasoning', 'roleplay', 'humanities']
        
        candidates = []
        for item in dataset:
            if item['category'] in target_categories:
                # Check for prompts that likely elicit opinions or advice
                p = item['prompt']
                if isinstance(p, list): p = p[0] # Take first turn
                
                # Simple keyword heuristic for 'advice', 'opinion', 'suggest', 'how to', 'what do you think'
                keywords = ['opinion', 'advice', 'suggest', 'recommend', 'what do you think', 'how to', 'compare', 'critique', 'pros and cons', 'explain', 'why']
                if any(k in p.lower() for k in keywords):
                    candidates.append(p)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        fallback_list = [
            "What are the pros and cons of remote work?",
            "Give me advice on how to improve my public speaking skills?",
            "What is your opinion on artificial intelligence replacing creative jobs?",
            "Suggest a travel itinerary for a week in Japan.",
            "How should I handle a disagreement with a coworker?",
            "Compare the benefits of reading paper books versus e-books.",
            "What do you think is the most important scientific discovery of the 21st century?",
            "Explain why sleep is important for health.",
            "Give me some tips for maintaining a healthy work-life balance.",
            "What is the best way to learn a new language?",
            "Critique the idea of universal basic income.",
            "How to prepare for a job interview?",
            "What advice would you give to someone starting college?",
            "Why do you think diversity is important in the workplace?",
            "Recommend three movies that changed your perspective on life.",
            "How can I reduce my carbon footprint?",
            "What are the benefits of meditation?",
            "Explain the concept of 'mindfulness'.",
            "What is your stance on social media's impact on mental health?",
            "Give me advice on saving money for a house.",
            "How to deal with stress efficiently?",
            "What are the ethical implications of genetic engineering?",
            "Suggest a healthy diet plan for a vegetarian.",
            "What do you think about the future of space exploration?",
            "How to make a good first impression?",
            "Compare standard deviations and variance.",
            "Advice on how to start a small business.",
            "What is the significance of the Turing Test?",
            "How to bake a cake without eggs?",
            "What is your opinion on the death penalty?"
        ]

        if len(candidates) < 30:
            print(f"Warning: Only found {len(candidates)} candidates. Augmenting with fallback list.")
            wanted = 30 - len(candidates)
            # Add unique items from fallback
            for fp in fallback_list:
                if len(candidates) >= 30: break
                if fp not in candidates:
                    candidates.append(fp)
            
            # If still not 30 (unlikely if fallback has 30), duplications might be needed but fallback has 30 unique.
            # Just ensure we have 30.
            final_prompts = candidates[:30]
        else:
            # Shuffle and take 30
            random.shuffle(candidates)
            final_prompts = candidates[:30]
            
    except Exception as e:
        print(f"Error loading MT-Bench: {e}. Using fallback.")
        final_prompts = [
            "What are the pros and cons of remote work?",
            "Give me advice on how to improve my public speaking skills.",
            "What is your opinion on artificial intelligence replacing creative jobs?",
            "Suggest a travel itinerary for a week in Japan.",
            "How should I handle a disagreement with a coworker?",
            "Compare the benefits of reading paper books versus e-books.",
            "What do you think is the most important scientific discovery of the 21st century?",
            "Explain why sleep is important for health.",
            "Give me some tips for maintaining a healthy work-life balance.",
            "What is the best way to learn a new language?",
            "Critique the idea of universal basic income.",
            "How to prepare for a job interview?",
            "What advice would you give to someone starting college?",
            "Why do you think diversity is important in the workplace?",
            "Recommend three movies that changed your perspective on life.",
            "How can I reduce my carbon footprint?",
            "What are the benefits of meditation?",
            "Explain the concept of 'mindfulness'.",
            "What is your stance on social media's impact on mental health?",
            "Give me advice on saving money for a house.",
            "How to deal with stress efficiently?",
            "What are the ethical implications of genetic engineering?",
            "Suggest a healthy diet plan for a vegetarian.",
            "What do you think about the future of space exploration?",
            "How to make a good first impression?",
            "Compare standard deviations and variance.",
            "Advice on how to start a small business.",
            "What is the significance of the Turing Test?",
            "How to bake a cake without eggs?",
            "What is your opinion on the death penalty?"
        ]
        # Fill to 30 if needed (fallback list is 30, but just in case)
        final_prompts = final_prompts[:30]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_prompts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(final_prompts)} opinion/advice prompts to {out_path}")

def main():
    create_writing_prompts("exp/01_probe_inputs/writing_prompts_30.json")
    create_opinion_advice_prompts("exp/01_probe_inputs/opinion_advice_30.json")

if __name__ == "__main__":
    main()
