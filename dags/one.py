import re
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# STEP 1: Load and Clean Data
# ----------------------------
def load_data(**kwargs):
    
    df = pd.read_csv(
        "/opt/airflow/data/one.csv",
        sep="|",
        engine="python",
        encoding="utf-8",
        quoting=3
    )
    
    df.columns = ["text", "label"]

    # Clean labels to match our expected list
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    valid_labels = ["negative", "neutral", "positive"]
    df = df[df["label"].isin(valid_labels)]

    print(f"Total valid rows loaded: {len(df)}")
    
    kwargs["ti"].xcom_push(key="texts", value=df["text"].tolist())
    kwargs["ti"].xcom_push(key="labels", value=df["label"].tolist())

# ----------------------------
# STEP 2: Load Model Name
# ----------------------------
def load_model(**kwargs):
    #
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
    kwargs["ti"].xcom_push(key="model_name", value=model_name)

# ----------------------------
# STEP 3: Run Predictions
# ----------------------------
def run_predictions(**kwargs):
    ti = kwargs["ti"]
    texts = ti.xcom_pull(key="texts", task_ids="load_data")
    model_name = ti.xcom_pull(key="model_name", task_ids="load_model")

    
    model = pipeline("sentiment-analysis", model=model_name, device=-1)

    preds = model(
        texts,
        truncation=True,
        max_length=256,
        padding=True,
        batch_size=8
    )

    
    predictions = [p["label"].lower() for p in preds]
    ti.xcom_push(key="predictions", value=predictions)

# ----------------------------
# STEP 4: Evaluate (THE FIX IS HERE)
# ----------------------------
def evaluate(**kwargs):
    ti = kwargs["ti"]
    
   
    current_task_id = ti.task_id
    task_num = current_task_id.split("_")[-1]
    pred_task_id = f"run_predictions_{task_num}"

    y_true = ti.xcom_pull(key="labels", task_ids="load_data")
    y_pred = ti.xcom_pull(key="predictions", task_ids=pred_task_id)


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
    dag_id="one",
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

    for i in range(1,101):
        t_pred = PythonOperator(
            task_id=f"run_predictions_{i}",
            python_callable=run_predictions,
        )

        t_eval = PythonOperator(
            task_id=f"evaluate_model_{i}",
            python_callable=evaluate,
        )

       
        t1 >> t2 >> t_pred >> t_eval