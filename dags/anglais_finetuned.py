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
    base_model_name = kwargs["ti"].xcom_pull(key="base_model", task_ids="load_model")
    ft_model_name = kwargs["ti"].xcom_pull(key="ft_model", task_ids="load_model")
    
    # Ensure text is cast to string
    safe_texts = [str(t) for t in texts]

    # --- A) BASE MODEL INFERENCE ---
    print(f"Deploying BASE MODEL ({base_model_name}) pipeline...")
    base_pipeline = pipeline("sentiment-analysis", model=base_model_name, device=-1)
    
    base_preds = base_pipeline(safe_texts, truncation=True, max_length=512)
    base_predictions = [p["label"].lower() for p in base_preds]

    # --- B) FINE-TUNED MODEL INFERENCE ---
    print(f"Deploying FINE-TUNED MODEL ({ft_model_name}) pipeline...")
    ft_pipeline = pipeline("sentiment-analysis", model=ft_model_name, tokenizer=ft_model_name, device=-1)
    
    ft_preds = ft_pipeline(safe_texts, truncation=True, max_length=512)

    # Model already returns "positive", "negative", "neutral" directly
    # Filter out any "irrelevant" predictions along with their corresponding texts
    valid_labels = {"positive", "negative", "neutral"}
    
    filtered = [
        (p["label"].lower(), label)
        for p, label in zip(ft_preds, kwargs["ti"].xcom_pull(key="labels", task_ids="load_data"))
        if p["label"].lower() in valid_labels
    ]
    
    ft_predictions = [item[0] for item in filtered]
    filtered_labels = [item[1] for item in filtered]

    kwargs["ti"].xcom_push(key="base_predictions", value=base_predictions)
    kwargs["ti"].xcom_push(key="ft_predictions", value=ft_predictions)
    kwargs["ti"].xcom_push(key="filtered_labels", value=filtered_labels)
    
    dropped = len(safe_texts) - len(ft_predictions)
    print(f"Dropped {dropped} 'irrelevant' predictions.")
    print("Inference completed for both candidate models.")

# ----------------------------
# STEP 4: Evaluate benchmark
# ----------------------------
def evaluate(**kwargs):
    y_true = kwargs["ti"].xcom_pull(key="labels", task_ids="load_data")
    base_pred = kwargs["ti"].xcom_pull(key="base_predictions", task_ids="run_predictions")
    
    ft_pred = kwargs["ti"].xcom_pull(key="ft_predictions", task_ids="run_predictions")
    ft_true = kwargs["ti"].xcom_pull(key="filtered_labels", task_ids="run_predictions")  # use filtered labels

    # Base model uses original y_true
    b_acc = accuracy_score(y_true, base_pred)
    b_prec = precision_score(y_true, base_pred, average="weighted", zero_division=0)
    b_rec = recall_score(y_true, base_pred, average="weighted", zero_division=0)
    b_f1 = f1_score(y_true, base_pred, average="weighted", zero_division=0)

    # Fine-tuned model uses filtered labels
    ft_acc = accuracy_score(ft_true, ft_pred)
    ft_prec = precision_score(ft_true, ft_pred, average="weighted", zero_division=0)
    ft_rec = recall_score(ft_true, ft_pred, average="weighted", zero_division=0)
    ft_f1 = f1_score(ft_true, ft_pred, average="weighted", zero_division=0)
    ...

    # Structured CLI Output
    print("\n" + "="*5 + " BASE MODEL " + "="*5)
    print(f"Accuracy: {b_acc:.4f}")
    print(f"Precision (weighted): {b_prec:.4f}")
    print(f"Recall (weighted): {b_rec:.4f}")
    print(f"F1-score (weighted): {b_f1:.4f}")

    print("\n" + "="*5 + " FINE-TUNED MODEL " + "="*5)
    print(f"Accuracy: {ft_acc:.4f}")
    print(f"Precision (weighted): {ft_prec:.4f}")
    print(f"Recall (weighted): {ft_rec:.4f}")
    print(f"F1-score (weighted): {ft_f1:.4f}")

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
