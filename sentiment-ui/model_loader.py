import logging
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

device = "cuda"

# ==========================================
# 1. Sentiment Classifier (local fine-tune)
# ==========================================
SENTIMENT_MODEL_PATH = "/app/model"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model.to(device)
sentiment_model.eval()

id2label = sentiment_model.config.id2label

# ==========================================
# 2. OpenAI config
# ==========================================
OPENAI_API_KEY = "my key"
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = "gpt-4o"

def _openai_chat(messages: list, temperature=0.6, max_tokens=300) -> str:
    res = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()


# ==========================================
# Inference functions
# ==========================================

@torch.no_grad()
def predict(text: str) -> dict:
    inputs = sentiment_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    confidence, predicted_class_idx = torch.max(probs, dim=0)
    label = id2label[predicted_class_idx.item()]
    scores = {id2label[i]: round(prob.item() * 100, 2) for i, prob in enumerate(probs)}

    return {
        "label": label,
        "confidence": round(confidence.item() * 100, 2),
        "scores": scores,
    }


def generate_opinion(text: str) -> dict:
    result = predict(text)
    label      = result["label"]
    confidence = result["confidence"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a sentiment analysis expert specializing in Tunisian dialect, "
                "Arabizi, Arabic, and French code-switching. Be analytical and specific."
            )
        },
        {
            "role": "user",
            "content": (
                f"This text was classified as {label} sentiment with {confidence}% confidence.\n"
                f"Text: \"{text}\"\n\n"
                f"Write a concise 2-3 sentence analytical opinion explaining:\n"
                f"- Which specific words or phrases drive the {label} sentiment\n"
                f"- Any dialect, slang, or code-switching that influenced the score\n"
                f"- Whether {confidence}% confidence seems appropriate\n\n"
                f"Be direct and specific."
            )
        }
    ]

    result["opinion"] = _openai_chat(messages, temperature=0.4, max_tokens=180)
    return result


def chat(history: list) -> dict:
    reply = _openai_chat(history, temperature=0.6, max_tokens=300)
    return {"reply": reply}
