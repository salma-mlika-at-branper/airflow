from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

def load_data(**kwargs):
    """
    Validates and paths to the individual datasets.
    """
    print("Loading data...")
    data_paths = [
        "/opt/airflow/data/multilang.csv",
        "/opt/airflow/data/tunisainone.csv",
        
    ]
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Data file not found at {path}")
    
    # We pass these paths to the next task implicitly or just return them
    return data_paths

def combine_data(**kwargs):
    """
    Combines the datasets into multilingual_dataset.csv
    """
    print("Combining data...")
    
    # Load each dataset
    try:
        multilang_df = pd.read_csv("/opt/airflow/data/multilang.csv")
    except Exception as e:
        print(f"Failed to load multilang: {e}")
        multilang_df = pd.DataFrame()
        
    try:
        one_df = pd.read_csv("/opt/airflow/data/tunisainone.csv")
    except Exception as e:
        print(f"Failed to load tunisainone: {e}")
        one_df = pd.DataFrame()
        
    
        
    # Combine them
    combined = pd.concat([multilang_df, one_df], ignore_index=True)
    
    if combined.empty:
        raise ValueError("Combined dataset is empty. Cannot proceed.")
        
    # Data Cleaning and Formatting
    combined = combined.dropna(subset=['text', 'label'])
    combined['label'] = combined['label'].astype(str).str.strip().str.lower()
    combined['text'] = combined['text'].astype(str)
    
    # Remove extremely short texts
    combined = combined[combined['text'].str.len() >= 5]
    
    # Drop duplicates
    combined = combined.drop_duplicates(subset=['text'])
    
    out_path = "/opt/airflow/data/multilingual_dataset.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    combined.to_csv(out_path, index=False)
    print(f"Saved combined dataset to {out_path} with {len(combined)} records.")
    return out_path

def load_model(**kwargs):
    """
    Downloads and prepares the model locally so the training task can use it.
    """
    print("Loading model...")
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    model_path = "/opt/airflow/models/pretrained_sentiment"
    
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
    
    print(f"Downloading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3, 
        ignore_mismatched_sizes=True
    )
    model.save_pretrained(model_path)
    
    return model_path

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1_weighted": f1}

def train_model(**kwargs):
    """
    Fine-tunes the model on the combined dataset.
    """
    print("Training model...")
    data_path = "/opt/airflow/data/multilingual_dataset.csv"
    model_path = "/opt/airflow/models/pretrained_sentiment"
    out_dir = "/opt/airflow/models/sup_train_tunisain"
    
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    label_map = {
        "positive": 0,
        "negative": 1,
        "neutral": 2
    }
    
    # Filter valid labels
    initial_len = len(df)
    df = df[df['label'].isin(label_map.keys())]
    print(f"Removed {initial_len - len(df)} records with invalid labels.")
    
    df['label'] = df['label'].map(label_map)
    df = df[['text', 'label']]
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)
        
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    id2label = {v: k for k, v in label_map.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,
        id2label=id2label,
        label2id=label_map,
        ignore_mismatched_sizes=True
    )
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3.0,             
        per_device_train_batch_size=16,   
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-6,               
        load_best_model_at_end=True,      
        metric_for_best_model="f1_weighted",
        warmup_ratio=0.1,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,     
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Model saved to {out_dir}")
    
    return eval_metrics

with DAG(
    dag_id="sup_train_tunisain_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'sentiment', 'tunisian', 'multilingual']
) as dag:
    
    task_load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    task_combine_data = PythonOperator(
        task_id="combine__data",
        python_callable=combine_data,
    )
    
    task_load_model = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
    )
    
    task_train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    # Dependencies
    task_load_data >> task_combine_data >> task_train_model
    task_load_model >> task_train_model
