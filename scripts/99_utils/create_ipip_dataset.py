import json
import os

def main():
    # IPIP-50 Items (Big Five Factor Markers)
    # Source: https://ipip.ori.org/newBigFive5broadKey.htm
    # (+) keyed -> Positive, (-) keyed -> Negative (Reverse)
    # We use all of them as prompts.
    
    items = {
        "Extraversion": [
            "I am the life of the party.",
            "I feel little concern for others.",
            "I am always prepared.",
            "I get stressed out easily.",
            "I have a rich vocabulary.",
            "I don't talk a lot.",
            "I am interested in people.", 
            "I leave my belongings around.",
            "I am relaxed most of the time.",
            "I have difficulty understanding abstract ideas.",
            "I feel comfortable around people.",
            "I insult people.",
            "I pay attention to details.",
            "I worry about things.",
            "I have a vivid imagination.",
            "I keep in the background.",
            "I sympathize with others' feelings.",
            "I make a mess of things.",
            "I seldom feel blue.",
            "I am not interested in abstract ideas.",
            "I start conversations.",
            "I am not interested in other people's problems.",
            "I get chores done right away.",
            "I am easily disturbed.",
            "I have excellent ideas.",
            "I have little to say.",
            "I have a soft heart.",
            "I often forget to put things back in their proper place.",
            "I get upset easily.",
            "I do not have a good imagination.",
            "I talk to a lot of different people at parties.",
            "I am not really interested in others.",
            "I like order.",
            "I change my mood a lot.",
            "I am quick to understand things.",
            "I don't like to draw attention to myself.",
            "I take time out for others.",
            "I shirk my duties.",
            "I have frequent mood swings.",
            "I use difficult words.",
            "I don't mind being the center of attention.",
            "I feel others' emotions.",
            "I follow a schedule.",
            "I get irritated easily.",
            "I spend time reflecting on things.",
            "I am quiet around strangers.",
            "I make people feel at ease.",
            "I am exacting in my work.",
            "I often feel blue.",
            "I am full of ideas."
        ]
    }
    
    # IPIP 50 items (Mixed list to ensure we have 50 unique items)
    # The above dict structure was incomplete for mapping, so let's use the standard flat list
    # mapped to traits if needed, but for json list output we just need the prompts.
    # Here is the clean list of 50 items.
    
    ipip_50_items = [
        # Extraversion
        "I am the life of the party.",
        "I feel comfortable around people.",
        "I start conversations.",
        "I talk to a lot of different people at parties.",
        "I don't mind being the center of attention.",
        "I don't talk a lot.", # Reverse
        "I keep in the background.", # Reverse
        "I have little to say.", # Reverse
        "I don't like to draw attention to myself.", # Reverse
        "I am quiet around strangers.", # Reverse
        
        # Agreeableness
        "I am interested in people.",
        "I sympathize with others' feelings.",
        "I have a soft heart.",
        "I take time out for others.",
        "I feel others' emotions.",
        "I make people feel at ease.",
        "I am not interested in other people's problems.", # Reverse
        "I insult people.", # Reverse
        "I am not really interested in others.", # Reverse
        "I feel little concern for others.", # Reverse
        
        # Conscientiousness
        "I am always prepared.",
        "I pay attention to details.",
        "I get chores done right away.",
        "I like order.",
        "I follow a schedule.",
        "I am exacting in my work.",
        "I leave my belongings around.", # Reverse
        "I make a mess of things.", # Reverse
        "I often forget to put things back in their proper place.", # Reverse
        "I shirk my duties.", # Reverse
        
        # Neuroticism (Emotional Stability)
        "I get stressed out easily.",
        "I worry about things.",
        "I am easily disturbed.",
        "I get upset easily.",
        "I change my mood a lot.",
        "I have frequent mood swings.",
        "I get irritated easily.",
        "I often feel blue.",
        "I am relaxed most of the time.", # Reverse
        "I seldom feel blue.", # Reverse
        
        # Openness
        "I have a rich vocabulary.",
        "I have a vivid imagination.",
        "I have excellent ideas.",
        "I am quick to understand things.",
        "I use difficult words.",
        "I spend time reflecting on things.",
        "I am full of ideas.",
        "I have difficulty understanding abstract ideas.", # Reverse
        "I am not interested in abstract ideas.", # Reverse
        "I do not have a good imagination." # Reverse
    ]
    
    prompts = []
    for item in ipip_50_items:
        # Format: "Consider the statement: '...' Desribe..."
        prompt = (
            "The following is a statement describing a personality trait.\n"
            "Imagine you are a human with a specific personality.\n"
            "Please write a short monologue (100 words) reacting to this statement from your perspective.\n"
            "Do NOT mention that you are an AI.\n\n"
            f"Statement: \"{item}\""
        )
        prompts.append(prompt)
        
    output_path = "exp/01_probe_inputs/ipip_50.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2)
        
    print(f"Created {output_path} with {len(prompts)} prompts.")

if __name__ == "__main__":
    main()
