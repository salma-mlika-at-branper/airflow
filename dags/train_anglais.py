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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1_weighted": f1}

# ----------------------------
# STEP 1: Load and preprocess data
# ----------------------------
def load_data(**kwargs):
    data_path = "/opt/airflow/data/train-anglais.csv"
    
    print(f"Loading data from {data_path}...")
    col_names = ["id", "game_title", "sentiment_label", "tweet_text"]
    df = pd.read_csv(data_path, header=None, names=col_names)
    
    # Handle missing/empty texts safely by dropping NaNs
    df = df.dropna(subset=['tweet_text', 'sentiment_label'])
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.strip()
    
    # Keep only Positive, Negative, Neutral. Drop Irrelevant.
    label_map = {
        "Positive": 0,
        "Negative": 1,
        "Neutral": 2
    }
    
    df = df[df['sentiment_label'].isin(label_map.keys())]
    df['label'] = df['sentiment_label'].map(label_map)
    df = df[['tweet_text', 'label']]
    
    # Train / Val Split (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Push paths to XCom instead of raw DataFrames
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
    # Base XLM-Roberta model per requirement
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
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
    
    # Disk caching for tokenized variables to avoid repeating the tokenization loop
    cache_dir_train = "/opt/airflow/data/tokenized_train_cache_roberta"
    cache_dir_val = "/opt/airflow/data/tokenized_val_cache_roberta"

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
            # Dynamic padding via collator relies on tokenization producing varied lengths up to standard max_lengths
            texts = [str(x) if x is not None and not pd.isna(x) else "" for x in examples['tweet_text']]
            return tokenizer(texts, truncation=True, max_length=128)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        train_dataset.save_to_disk(cache_dir_train)
        val_dataset.save_to_disk(cache_dir_val)

    # Implement dynamic batch padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Loading model {model_name} (num_labels=3)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3, 
        ignore_mismatched_sizes=True
    )
    
    # 3 epochs, batch 16, lr 1e-5 to avoid pre-trained weight destruction. Output strictly CPU.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3.0,             
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,               
        load_best_model_at_end=True,      
        metric_for_best_model="f1_weighted",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        no_cuda=True,                     # Strictly limit runtime to CPU allocation bounds natively
        fp16=False,                       # Ensure Mixed-precision floating point logic is stripped for standalone CPU iteration
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,     # Map the padding logic
    )
    
    print("Starting Model Training...")
    trainer.train()
    
    print("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    
    print(f"Saving final trained model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
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
        python_callable=load_data,
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
