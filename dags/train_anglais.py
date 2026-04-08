from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch

# Global DEV mode flag for faster pipeline iterations
DEV_MODE = True

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1_weighted": f1}

# ----------------------------
# STEP 1: Load and preprocess data
# ----------------------------
def load_and_preprocess_data(**kwargs):
    # Adjust this path based on your Airflow Docker mount
    data_path = "/opt/airflow/data/train-anglais.csv"
    
    print(f"Loading data from {data_path}...")
    col_names = ["id", "game_title", "sentiment_label", "tweet_text"]
    df = pd.read_csv(data_path, header=None, names=col_names)
    
    # Drop rows where 'tweet_text' or 'sentiment_label' is null
    df = df.dropna(subset=['tweet_text', 'sentiment_label'])
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.strip()
    
    label_map = {
        "Positive": 0,
        "Negative": 1,
        "Neutral": 2,
        "Irrelevant": 3
    }
    
    df = df[df['sentiment_label'].isin(label_map.keys())]
    df['label'] = df['sentiment_label'].map(label_map)
    df = df[['tweet_text', 'label']]
    
    if DEV_MODE:
        print("DEV_MODE is True: Subsampling dataset to max 3000 rows for fast testing.")
        df = df.sample(n=min(3000, len(df)), random_state=42)
    
    # Train / Val Split (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_path = "/opt/airflow/data/processed_train.csv"
    val_path = "/opt/airflow/data/processed_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    kwargs["ti"].xcom_push(key="train_data_path", value=train_path)
    kwargs["ti"].xcom_push(key="val_data_path", value=val_path)

# ----------------------------
# STEP 2: Configure model parameters
# ----------------------------
def load_model_config(**kwargs):
    # distilbert-base-uncased is much faster / lighter than roberta
    model_name = "distilbert-base-uncased"
    output_dir = "/opt/airflow/models/sentiment_model"
    
    os.makedirs(output_dir, exist_ok=True)
    
    kwargs["ti"].xcom_push(key="model_name", value=model_name)
    kwargs["ti"].xcom_push(key="output_dir", value=output_dir)

# ----------------------------
# STEP 3: Train the model
# ----------------------------
def run_training(**kwargs):
    train_path = kwargs["ti"].xcom_pull(key="train_data_path", task_ids="load_data")
    val_path = kwargs["ti"].xcom_pull(key="val_data_path", task_ids="load_data")
    model_name = kwargs["ti"].xcom_pull(key="model_name", task_ids="load_model_config")
    output_dir = kwargs["ti"].xcom_pull(key="output_dir", task_ids="load_model_config")

    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Define cache directories conditionally based on DEV_MODE
    dev_suffix = "_dev" if DEV_MODE else ""
    cache_dir_train = f"/opt/airflow/data/tokenized_train_cache{dev_suffix}"
    cache_dir_val = f"/opt/airflow/data/tokenized_val_cache{dev_suffix}"

    # Verify if caching mechanism hit
    if os.path.exists(cache_dir_train) and os.path.exists(cache_dir_val):
        print("Caching HIT! Loading tokenized datasets directly from disk...")
        train_dataset = load_from_disk(cache_dir_train)
        val_dataset = load_from_disk(cache_dir_val)
    else:
        print("Caching MISS! Reading CSV files and tokenizing datasets...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        def tokenize_function(examples):
            texts = [str(x) if x is not None and not pd.isna(x) else "" for x in examples['tweet_text']]
            # Only truncation applied, no max_length padding. Let the DataCollator handle padded variations
            return tokenizer(texts, truncation=True, max_length=128)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Save pre-tokenized states immediately to disk buffer
        train_dataset.save_to_disk(cache_dir_train)
        val_dataset.save_to_disk(cache_dir_val)

    # Create dynamic padding batch collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Loading model {model_name} (num_labels=4)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4, 
        ignore_mismatched_sizes=True
    )
    
    # strictly optimized CPU constraints
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1.0,           # Lower epoch for DEV
        per_device_train_batch_size=32, # Optimize inference throughput
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        no_cuda=True,                 # Explicit manual override to remove all CUDA searches
        fp16=False,                   # Do not attempt Mixed Precision conversions natively mapping Nvidia Tensor cores
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # Use dynamic padding instance 
    )
    
    print("Starting Model Training...")
    trainer.train()
    
    print("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    
    print(f"Saving final trained model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Store metrics for the next task
    kwargs["ti"].xcom_push(key="eval_metrics", value=eval_metrics)

# ----------------------------
# STEP 4: Evaluate results
# ----------------------------
def evaluate_model(**kwargs):
    eval_metrics = kwargs["ti"].xcom_pull(key="eval_metrics", task_ids="run_training")
    
    print("------------------------")
    print("Final Evaluation Metrics (After Training):")
    for key, value in eval_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("------------------------")

# ----------------------------
# DAG definition
# ----------------------------
with DAG(
    dag_id="train_anglais",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=load_and_preprocess_data,
    )

    t2 = PythonOperator(
        task_id="load_model_config",
        python_callable=load_model_config,
    )

    t3 = PythonOperator(
        task_id="run_training",
        python_callable=run_training,
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t1 >> t2 >> t3 >> t4
