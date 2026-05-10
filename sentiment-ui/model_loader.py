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
# 2. Ollama config
# ==========================================
OLLAMA_HOSTS = [
    "http://host.docker.internal:11434",
    "http://172.17.0.1:11434",
]
OLLAMA_MODEL = "aya:8b:latest"


def _ollama_post(path: str, payload: dict) -> dict:
    last_err = None
    for host in OLLAMA_HOSTS:
        try:
            res = requests.post(f"{host}{path}", json=payload, timeout=120)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            last_err = e
            logger.warning(f"Ollama host {host} failed: {e}")
    raise RuntimeError(f"All Ollama hosts failed. Last error: {last_err}")


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

    prompt = (
        f"You are an AI assistant specialized in the Tunisian dialect. You must understand and communicate fluently in 'Derja' (Tunisian Arabic and latin), including the frequent mixing of French and English words (Code-switching). Do not correct the user to formal Arabic. Reply naturally as a local Tunisian would.\n\n"
        f"A text was classified as **{label}** sentiment with {confidence}% confidence.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Write a concise 2-3 sentence analytical opinion:\n"
        f"- Which specific words/phrases drive the {label} sentiment\n"
        f"- Any dialect, slang, or code-switching that influenced the score\n"
        f"- Whether {confidence}% confidence seems appropriate\n\n"
        f"Be direct. Do not restate the task."
    )

    data = _ollama_post("/api/generate", {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 180},
    })

    result["opinion"] = data.get("response", "").strip()
    return result


def chat(history: list) -> dict:
    data = _ollama_post("/api/chat", {
        "model": OLLAMA_MODEL,
        "messages": history,
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 300},
    })

    reply = data.get("message", {}).get("content", "").strip()
    return {"reply": reply}