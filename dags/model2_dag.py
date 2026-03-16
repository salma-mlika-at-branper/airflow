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
    df = pd.read_csv("/opt/airflow/data/test.csv")
    print(df.head())
    # push texts and original sentiment labels directly
    kwargs["ti"].xcom_push(key="texts", value=df["text"].tolist())
    kwargs["ti"].xcom_push(key="labels", value=df["sentiment"].tolist())

# ----------------------------
# STEP 2: Load pretrained model (store only model name)
# ----------------------------
def load_model(**kwargs):
    model_name = "mervp/SentimentBERT"
    kwargs["ti"].xcom_push(key="model_name", value=model_name)

# ----------------------------
# STEP 3: Run predictions
# ----------------------------
def run_predictions(**kwargs):
    texts = kwargs["ti"].xcom_pull(key="texts", task_ids="load_data")
    model_name = kwargs["ti"].xcom_pull(key="model_name", task_ids="load_model")

    model = pipeline("sentiment-analysis", model=model_name)
    preds = model(texts)

    # SentimentBERT outputs labels directly: "positive", "neutral", "negative"
    predictions = [p["label"].lower() for p in preds]
    kwargs["ti"].xcom_push(key="predictions", value=predictions)

# ----------------------------
# STEP 4: Evaluate benchmark
# ----------------------------
def evaluate(**kwargs):
    y_true = kwargs["ti"].xcom_pull(key="labels", task_ids="load_data")
    y_pred = kwargs["ti"].xcom_pull(key="predictions", task_ids="run_predictions")

    acc = accuracy_score(y_true, y_pred)
    print(f"Benchmark Accuracy: {acc}")

    acc = accuracy_score(y_true, y_pred) 
    prec = precision_score(y_true, y_pred, average="weighted") 
    rec = recall_score(y_true, y_pred, average="weighted") 
    f1 = f1_score(y_true, y_pred, average="weighted") 
    print(f"Benchmark Accuracy: {acc:.4f}") 
    print(f"Precision (weighted): {prec:.4f}") 
    print(f"Recall (weighted): {rec:.4f}") 
    print(f"F1-score (weighted): {f1:.4f}")

# ----------------------------
# DAG definition
# ----------------------------
with DAG(
    dag_id="model2",
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
