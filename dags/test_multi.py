from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from transformers import pipeline
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# STEP 1: Load real data
# ----------------------------
def load_data(**kwargs):
    df = pd.read_csv("/opt/airflow/data/multilang.csv")
    
    # Safely assign columns dynamically ensuring it doesn't break if there are only 2 columns
    col_target = "sentiment" if "sentiment" in df.columns else "label"
    if col_target not in df.columns:
        df.columns = ["text", "label"]
        col_target = "label"

    df[col_target] = df[col_target].astype(str).str.strip().str.lower()
    
    # Keep only the 3 labels present in anglais.csv
    valid_labels = {"positive", "negative", "neutral"}
    df = df[df[col_target].isin(valid_labels)]

    print(f"Label distribution:\n{df[col_target].value_counts()}")

    # Pushing lists to Xcom (Warning: Huge datasets may overflow Airflow XCom Database)
    kwargs["ti"].xcom_push(key="texts", value=df["text"].tolist())
    kwargs["ti"].xcom_push(key="labels", value=df[col_target].tolist())


# ----------------------------
# STEP 2: Load fine-tuned model path
# ----------------------------
def load_model(**kwargs):
    # Point to the fine-tuned model
    model_path = "/opt/airflow/models/twitter_sentiment_finetuned"
    kwargs["ti"].xcom_push(key="model_path", value=model_path)


# ----------------------------
# STEP 3: Run predictions
# ----------------------------
def run_predictions(**kwargs):
    texts = kwargs["ti"].xcom_pull(key="texts", task_ids="load_data")
    model_path = kwargs["ti"].xcom_pull(key="model_path", task_ids="load_model")

    device = 0 if torch.cuda.is_available() else -1
    print(f"Running inference on {'GPU' if device == 0 else 'CPU'}...")

    # Load your fine-tuned model
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path,
        truncation=True,
        max_length=128,  # Match training tokenization
        device=device
    )

    print(f"Evaluating {len(texts)} samples...")
    preds = sentiment_pipeline(texts, batch_size=32)
    predictions = [p["label"].lower() for p in preds]
    
    kwargs["ti"].xcom_push(key="predictions", value=predictions)


# ----------------------------
# STEP 4: Evaluate
# ----------------------------
def evaluate(**kwargs):
    y_true = kwargs["ti"].xcom_pull(key="labels", task_ids="load_data")
    y_pred = kwargs["ti"].xcom_pull(key="predictions", task_ids="run_predictions")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("=== Fine-tuned Model Evaluation on multilang.csv ===")
    print(f"Accuracy:             {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted):    {rec:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")


# ----------------------------
# DAG definition
# ----------------------------
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="multi_finetuned_eval",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["nlp", "eval"]
) as dag:

    t1 = PythonOperator(task_id="load_data",       python_callable=load_data)
    t2 = PythonOperator(task_id="load_model",      python_callable=load_model)
    t3 = PythonOperator(task_id="run_predictions", python_callable=run_predictions)
    t4 = PythonOperator(task_id="evaluate_model",  python_callable=evaluate)

    t1 >> t2 >> t3 >> t4
