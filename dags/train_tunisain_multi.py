from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil
from datasets import Dataset
import torch
torch.cuda.empty_cache()
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import logging

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

# --- File Paths ---
DATA_PATH = "/opt/airflow/data/merged_data.csv"
TRAIN_DATA_PATH = "/opt/airflow/data/train.parquet"
TEST_DATA_PATH = "/opt/airflow/data/test.parquet"
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BASE_MODEL_DIR = "/opt/airflow/models/base_model"
TEMP_TRAIN_DIR = "/opt/airflow/models/temp_training"
FINAL_MODEL_DIR = "/opt/airflow/models/pretrained_model/"

def load_data(**kwargs):
    """
    1. Load Dataset & 3. Data Preprocessing
    """
    logging.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    df = df[['text', 'label']]
    
    # 3. Data Preprocessing - Map labels
    label_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df = df[df['label'].isin(label_mapping.keys())]
    df['label'] = df['label'].map(label_mapping)
    
    # Split dataset into train/test (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logging.info(f"Train set size: {len(train_df)}")
    logging.info(f"Test set size: {len(test_df)}")
    
    # Save intermediate files for train_model to pick up
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)
    logging.info("Saved train and test datasets.")

def load_model(**kwargs):
    """
    2. Load Model
    """
    logging.info(f"Downloading/Loading base model from {MODEL_NAME}")
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    
    # Load model with num_labels=3
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.save_pretrained(BASE_MODEL_DIR)
    logging.info(f"Model and tokenizer saved to {BASE_MODEL_DIR}")

def compute_metrics(eval_pred):
    """
    6. Metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1
    }

def train_model(**kwargs):
    """
    4. Tokenization & 5. Training
    """
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load tokenizer and model from local cache
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    
    def tokenize_function(examples):
        # 4. Max Length 128, Truncation applied
        return tokenizer(examples['text'], padding=False, truncation=True, max_length=128)
        
    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # 4. Padding is dynamically applied during batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_DIR, num_labels=3)
    
    # 5. TrainingArguments config
    training_args = TrainingArguments(
        output_dir=TEMP_TRAIN_DIR,
        num_train_epochs=3,             
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        fp16=True,
        gradient_checkpointing=True,
        learning_rate=5e-5,               
        evaluation_strategy="epoch",      # Evaluate at each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,     
    )
    
    logging.info("Starting Training...")
    trainer.train()
    
    logging.info("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    
    # 6. Print metrics in logs
    logging.info("Final Evaluation Metrics:")
    for key, value in eval_metrics.items():
        logging.info(f"  {key}: {value}")
        
    # Save the trained model temporarily 
    trainer.save_model(TEMP_TRAIN_DIR)
    tokenizer.save_pretrained(TEMP_TRAIN_DIR)

def save_model(**kwargs):
    """
    7. Save Model
    """
    logging.info(f"Saving final trained model to {FINAL_MODEL_DIR}")
    
    # Create or overwrite the final directory
    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
        
    shutil.copytree(TEMP_TRAIN_DIR, FINAL_MODEL_DIR)
    logging.info("Model saved successfully.")

# --- DAG Definition ---
with DAG(
    dag_id="train_multilingual_sentiment_model",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'sentiment', 'trainer', 'transformers']
) as dag:
    
    task_load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    task_load_model = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
    )
    
    task_train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_save_model = PythonOperator(
        task_id="save_model",
        python_callable=save_model,
    )

    # DAG Structure Dependency
    task_load_data >> task_load_model >> task_train_model >> task_save_model
