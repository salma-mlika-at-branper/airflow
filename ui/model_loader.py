import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration
)

device = "cuda"

# ==========================================
# 1. Load the Sentiment Classifier (Local)
# ==========================================
# This loads directly from the local volume mount, NOT from Hugging Face
SENTIMENT_MODEL_PATH = "/app/model"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model.to(device)
sentiment_model.eval()

# Read the label mapping dynamically from the sentiment model's config
id2label = sentiment_model.config.id2label

# ==========================================
# 2. Load FLAN-T5-Base (Hugging Face)
# ==========================================
T5_MODEL_NAME = "google/flan-t5-base"
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
t5_model.to(device)
t5_model.eval()

@torch.no_grad()
def predict(text: str) -> dict:
    """Predicts sentiment for a given text."""
    inputs = sentiment_tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128
    )
    
    # Move all input tensors to GPU before inference
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    confidence, predicted_class_idx = torch.max(probs, dim=0)
    label = id2label[predicted_class_idx.item()]
    
    scores = {id2label[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probs)}
    
    return {
        "label": label,
        "confidence": round(confidence.item() * 100, 2),
        "scores": scores
    }

@torch.no_grad()
def generate_opinion(text: str) -> dict:
    """Runs predict(), then uses flan-t5-base to generate an opinion."""
    # 1. Run prediction first
    result = predict(text)
    
    # 2. Extract values for the prompt
    label = result["label"]
    confidence = result["confidence"]
    
    # 3. Format the specific prompt
    prompt = f"The following text expressed a {label} sentiment with {confidence}% confidence. Text: '{text}'. Write a short analytical opinion about the sentiment expressed."
    
    # 4. Tokenize for T5
    t5_inputs = t5_tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True,
        max_length=256
    )
    
    # Move all input tensors to GPU before inference
    t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
    
    # 5. Generate opinion
    outputs = t5_model.generate(
        **t5_inputs,
        max_new_tokens=150
    )
    
    # 6. Decode output
    opinion = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 7. Add generated text to result dict
    result["opinion"] = opinion
    
    return result
