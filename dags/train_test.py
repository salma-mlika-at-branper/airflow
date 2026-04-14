from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ----------------------------
# STEP 1: Load real data
# ----------------------------
def load_data(**kwargs):
    df = pd.read_csv("/opt/airflow/data/anglais.csv")
    df.columns = ["textID", "text", "sentiment"]
    print(df.head())

    # Normalize labels to title-case to match the fine-tuned model's id2label output
    # The fine-tuned model outputs: "Positive", "Negative", "Neutral", "Irrelevant"
    df["sentiment"] = df["sentiment"].str.strip().str.capitalize()

    # Keep only the 3 labels present in anglais.csv
    valid_labels = {"Positive", "Negative", "Neutral"}
    df = df[df["sentiment"].isin(valid_labels)]

    print(f"Label distribution:\n{df['sentiment'].value_counts()}")

    kwargs["ti"].xcom_push(key="texts", value=df["text"].tolist())
    kwargs["ti"].xcom_push(key="labels", value=df["sentiment"].tolist())


# ----------------------------
# STEP 2: Load fine-tuned model path
# ----------------------------
def load_model(**kwargs):
    # ✅ Point to YOUR fine-tuned model, not the base one
    model_path = "/opt/airflow/models/twitter_sentiment_finetuned"
    kwargs["ti"].xcom_push(key="model_path", value=model_path)


# ----------------------------
# STEP 3: Run predictions
# ----------------------------
def run_predictions(**kwargs):
    texts = kwargs["ti"].xcom_pull(key="texts", task_ids="load_data")
    model_path = kwargs["ti"].xcom_pull(key="model_path", task_ids="load_model")

    # Load your fine-tuned model
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path,
        truncation=True,
        max_length=128  # ✅ Match training tokenization
    )

    preds = sentiment_pipeline(texts, batch_size=16)
    predictions = [p["label"] for p in preds]
    kwargs["ti"].xcom_push(key="predictions", value=predictions)


# ----------------------------
# STEP 4: Evaluate
# ----------------------------
def evaluate(**kwargs):
    y_true = kwargs["ti"].xcom_pull(key="labels", task_ids="load_data")
    y_pred = kwargs["ti"].xcom_pull(key="predictions", task_ids="run_predictions")

    labels = ["Positive", "Negative", "Neutral"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", labels=labels)
    rec  = recall_score(y_true, y_pred, average="weighted", labels=labels)
    f1   = f1_score(y_true, y_pred, average="weighted", labels=labels)

    print("=== Fine-tuned Model Evaluation on anglais.csv ===")
    print(f"Accuracy:             {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted):    {rec:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")


# ----------------------------
# DAG definition
# ----------------------------
with DAG(
    dag_id="anglais_finetuned_eval",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id="load_data",        python_callable=load_data)
    t2 = PythonOperator(task_id="load_model",       python_callable=load_model)
    t3 = PythonOperator(task_id="run_predictions",  python_callable=run_predictions)
    t4 = PythonOperator(task_id="evaluate_model",   python_callable=evaluate)

    t1 >> t2 >> t3 >> t4