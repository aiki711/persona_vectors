import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_classifier():
    model_name = "KevSun/Personality_LM"
    texts = [
        "I love going to parties and meeting new people! It is so exciting to be the center of attention.",
        "I prefer to stay at home and read a book. I find social gatherings exhausting and loud.",
        "I always keep my desk organized and finish my work ahead of schedule.",
        "I often forget my appointments and leave my room in a mess.",
        "I sympathize with others' feelings and try to help those in need.",
        "I don't care about other people's problems; they should solve them themselves.",
        "I have a vivid imagination and enjoy abstract ideas.",
        "I am not interested in abstract ideas; I prefer practical and concrete things.",
        "I get stressed out easily and worry about everything.",
        "I am relaxed most of the time and don't worry much."
    ]
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    print("Labels:", model.config.id2label)
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    print("\n--- Scores (Logits) ---")
    for text, score in zip(texts, logits):
        print(f"Text: {text[:50]}...")
        # KevSun mapping: 0:Extraversion, 1:Neuroticism, 2:Agreeableness, 3:Conscientiousness, 4:Openness
        print(f"  Ext: {score[0]:.3f}, Neu: {score[1]:.3f}, Agr: {score[2]:.3f}, Con: {score[3]:.3f}, Opn: {score[4]:.3f}")

if __name__ == "__main__":
    test_classifier()
