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
OLLAMA_MODEL = "mistral:latest"


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

    messages = [
        {
            "role": "system",
            "content": (
                "Inti assistant Tounsi. Tfhem derja, arabizi, français, w english mzejin. "
                "Jaweb b nafs el logha mta3 el input — ki el user yekteb b derja, jaweb b derja. "
                "Ki yekteb b français, jaweb b français. Ki yekteb b english, jaweb b english. "
                "Ma t9addeche lil arabic alfussha. Tkellm naturally ki Tounsi 3adi."
            )
        },
        {
            "role": "user",
            "content": (
                f"El text hedha: \"{text}\"\n"
                f"El model 9al aliha {label} b confidence ta3 {confidence}%.\n\n"
                f"3tini interpretation 9sira (2-3 jmal):\n"
                f"- Chnoua el klemt elli khallew el sentiment {label}\n"
                f"- El confidence ta3 {confidence}% logique wela le9?\n"
                f"Jaweb b nafs el logha mta3 el text."
            )
        }
    ]

    data = _ollama_post("/api/chat", {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 180},
    })

    result["opinion"] = data.get("message", {}).get("content", "").strip()
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