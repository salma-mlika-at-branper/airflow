from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ----------------------------
# STEP 1: Load real data
# ----------------------------
def load_data(**kwargs):
    print("Loading test dataset...")
    df = pd.read_csv("/opt/airflow/data/anglais.csv")
    
    # Optional safety: rename columns if not already cleanly formatted
    df.columns = ["textID", "text", "sentiment"]

    # Drop missing values to prevent sklearn metric errors
    df = df.dropna(subset=['text', 'sentiment'])
 
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    # Map string labels to standardized lowercase sentiment strings
    label_map = {
        "negative": "negative",
        "positive": "positive",
        "neutral": "neutral"
    }
    df["sentiment"] = df["sentiment"].map(label_map)
    df = df.dropna(subset=['sentiment']) # drop rows that had unexpected labels
    
    texts = df["text"].tolist()
    labels = df["sentiment"].tolist()

    print(f"Dataset securely loaded. Shape: {len(texts)} entries.")
    # push texts and original sentiment labels directly
    kwargs["ti"].xcom_push(key="texts", value=texts)
    kwargs["ti"].xcom_push(key="labels", value=labels)

# ----------------------------
# STEP 2: Load pretrained model (store model names)
# ----------------------------
def load_model(**kwargs):
    # Store both the baseline and our new fine-tuned model path
    base_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    ft_model = "/opt/airflow/models/sentiment_model_finetuned"
    kwargs["ti"].xcom_push(key="base_model", value=base_model)
    kwargs["ti"].xcom_push(key="ft_model", value=ft_model)
    print("Model identities populated inside XCom cache...")
 
# ----------------------------
# STEP 3: Run predictions
# ----------------------------
def run_predictions(**kwargs):
    texts = kwargs["ti"].xcom_pull(key="texts", task_ids="load_data")
    labels = kwargs["ti"].xcom_pull(key="labels", task_ids="load_data")
    base_model_name = kwargs["ti"].xcom_pull(key="base_model", task_ids="load_model")
    ft_model_name = kwargs["ti"].xcom_pull(key="ft_model", task_ids="load_model")

    safe_texts = [str(t) for t in texts]
    valid_labels = {"positive", "negative", "neutral"}

    # --- BASE MODEL ---
    print(f"Deploying BASE MODEL ({base_model_name})...")
    base_pipeline = pipeline("sentiment-analysis", model=base_model_name, device=-1)
    base_preds_raw = base_pipeline(safe_texts, truncation=True, max_length=512)

    # --- BASE MODEL ---
print(f"Deploying BASE MODEL ({base_model_name})...")
base_pipeline = pipeline("sentiment-analysis", model=base_model_name, device=-1)
base_preds_raw = base_pipeline(safe_texts, truncation=True, max_length=512)

base_filtered = [
    (p["label"].lower(), l)
    for p, l in zip(base_preds_raw, labels)
    if p["label"].lower() in valid_labels
]

base_predictions = [x[0] for x in base_filtered]
base_true_labels = [x[1] for x in base_filtered]


# --- FINE-TUNED MODEL ---
print(f"Deploying FINE-TUNED MODEL ({ft_model_name})...")
ft_pipeline = pipeline("sentiment-analysis", model=ft_model_name, tokenizer=ft_model_name, device=-1)
ft_preds_raw = ft_pipeline(safe_texts, truncation=True, max_length=512)

ft_filtered = [
    (p["label"].lower(), l)
    for p, l in zip(ft_preds_raw, labels)
    if p["label"].lower() in valid_labels
]

    ft_predictions = [x[0] for x in ft_filtered]
    ft_true_labels = [x[1] for x in ft_filtered]
    base_predictions  = [x[0] for x in base_filtered]
    base_true_labels  = [x[1] for x in base_filtered]

    # - FINE-TUNED MODEL ---
    print(f"Deploying FINE-TUNED MODEL ({ft_model_name})...")
    ft_pipeline = pipeline("sentiment-analysis", model=ft_model_name,
                           tokenizer=ft_model_name, device=-1)
    ft_preds_raw = ft_pipeline(safe_texts, truncation=True, max_length=512)

    # Align ft predictions with labels (filter both together)
    ft_filtered = [
        (p["label"].lower(), l)
        for p, l in zip(ft_preds_raw, labels)
        if p["label"].lower() in valid_labels
    ]
    ft_predictions  = [x[0] for x in ft_filtered]
    ft_true_labels  = [x[1] for x in ft_filtered]

    kwargs["ti"].xcom_push(key="base_predictions",  value=base_predictions)
    kwargs["ti"].xcom_push(key="base_true_labels",  value=base_true_labels)
    kwargs["ti"].xcom_push(key="ft_predictions",    value=ft_predictions)
    kwargs["ti"].xcom_push(key="filtered_labels",   value=ft_true_labels)

    print(f"Base dropped: {len(safe_texts) - len(base_predictions)}")
    print(f"FT dropped:   {len(safe_texts) - len(ft_predictions)}")
# ----------------------------
# STEP 4: Evaluate benchmark
# ----------------------------
def evaluate(**kwargs):
    base_pred = kwargs["ti"].xcom_pull(key="base_predictions", task_ids="run_predictions")
    base_true = kwargs["ti"].xcom_pull(key="base_true_labels",  task_ids="run_predictions")
    ft_pred   = kwargs["ti"].xcom_pull(key="ft_predictions",    task_ids="run_predictions")
    ft_true   = kwargs["ti"].xcom_pull(key="filtered_labels",   task_ids="run_predictions")

    b_acc  = accuracy_score(base_true, base_pred)
    b_prec = precision_score(base_true, base_pred, average="weighted", zero_division=0)
    b_rec  = recall_score(base_true, base_pred, average="weighted", zero_division=0)
    b_f1   = f1_score(base_true, base_pred, average="weighted", zero_division=0)

    ft_acc  = accuracy_score(ft_true, ft_pred)
    ft_prec = precision_score(ft_true, ft_pred, average="weighted", zero_division=0)
    ft_rec  = recall_score(ft_true, ft_pred, average="weighted", zero_division=0)
    ft_f1   = f1_score(ft_true, ft_pred, average="weighted", zero_division=0)

    print("\n===== BASE MODEL =====")
    print(f"Accuracy:             {b_acc:.4f}")
    print(f"Precision (weighted): {b_prec:.4f}")
    print(f"Recall (weighted):    {b_rec:.4f}")
    print(f"F1-score (weighted):  {b_f1:.4f}")

    print("\n===== FINE-TUNED MODEL =====")
    print(f"Accuracy:             {ft_acc:.4f}")
    print(f"Precision (weighted): {ft_prec:.4f}")
    print(f"Recall (weighted):    {ft_rec:.4f}")
    print(f"F1-score (weighted):  {ft_f1:.4f}")

# ----------------------------
# DAG definition
# ----------------------------
with DAG(
    dag_id="anglais_fine_tuned",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    t2 = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
    )

    t3 = PythonOperator(
        task_id="run_predictions",
        python_callable=run_predictions,
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate,
    )

    t1 >> t2 >> t3 >> t4
