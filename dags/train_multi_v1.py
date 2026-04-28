"""
DAG for fine-tuning the twitter-xlm-roberta-base-sentiment model.
"""

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

def compute_metrics(eval_pred):
    """
    Computes accuracy and weighted F1 score for the model evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1_weighted": f1}


def load_clean_data(data_path, label_map):
    """
    Loads and cleans the dataset, yielding formatted train and validation datasets.
    """
    print(f"Loading data from {data_path}...")
    # The file seems to have a mixed separator, so we read it with a regex separator
    df = pd.read_csv(data_path, sep=',|\\t', engine='python', on_bad_lines='skip')
    
    if not df.empty:
        df.columns = df.columns.str.strip()
        
    if 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
        
    if 'label' not in df.columns:
        # if 'label' is still not in columns, try a different approach
        df = pd.read_csv(data_path, header=None, names=['text', 'label'], sep=',|\\t', engine='python', on_bad_lines='skip')

    initial_len = len(df)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.len() >= 5]
    df = df.drop_duplicates(subset=['text'])
    print(f"Removed {initial_len - len(df)} duplicate, null, or short rows.")
    
    initial_len_labels = len(df)
    df = df[df['label'].isin(label_map.keys())]
    print(f"Removed {initial_len_labels - len(df)} rows with incorrect/irrelevant labels.")
    
    df['label'] = df['label'].map(label_map)
    df = df[['text', 'label']]
    
    print("Splitting data into train and validation sets (80/20)...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def load_model_instance(model_name, label_map):
    """
    Loads the Hugging Face model, tokenizer, and data collator.
    """
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading AutoModelForSequenceClassification for {model_name} with num_labels=3...")
    id2label = {v: k for k, v in label_map.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=id2label,
        label2id=label_map
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return model, tokenizer, data_collator

def execute_training(model, train_dataset, val_dataset, data_collator, output_dir, kwargs):
    """
    Sets up TrainingArguments and runs the model custom training step.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("num_train_epochs", 3.0),             
        per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 16),   
        per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 16),
        evaluation_strategy=kwargs.get("evaluation_strategy", "epoch"),
        save_strategy=kwargs.get("save_strategy", "epoch"),
        learning_rate=kwargs.get("learning_rate", 5e-6),               
        load_best_model_at_end=True,      
        metric_for_best_model="f1_weighted",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        warmup_ratio=kwargs.get("warmup_ratio", 0.1),
        weight_decay=kwargs.get("weight_decay", 0.01),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,     
    )
    
    print("Starting Model Training...")
    trainer.train()
    
    return trainer

def execute_evaluation(trainer, output_dir, model, tokenizer):
    """
    Evaluates the model on validation data and saves the artifacts.
    """
    print("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    
    print("------------------------")
    print("Final Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("------------------------")
        
    print(f"Saving fine-tuned model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return eval_metrics

def main_pipeline(**kwargs):
    """
    Main orchestration function running the sequence of modularized components.
    """
    # 1. Retrieve config parameters
    data_path = kwargs.get("data_path", "/opt/airflow/data/mix3.csv")
    model_name = kwargs.get("model_name", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
    output_dir = kwargs.get("output_dir", "/opt/airflow/models/sentiment_model_v1")
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    # 2. Data Source Load & Clean
    train_dataset, val_dataset = load_clean_data(data_path, label_map)
    
    # 3. Model Loading
    model, tokenizer, data_collator = load_model_instance(model_name, label_map)
    
    # 4. Tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)
        
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Remove unused columns after tokenization
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names 
         if col not in ["input_ids", "attention_mask", "label"]]
    )
    val_dataset = val_dataset.remove_columns(
        [col for col in val_dataset.column_names 
         if col not in ["input_ids", "attention_mask", "label"]]
    )
    
    # 5. Training Stage
    trainer = execute_training(model, train_dataset, val_dataset, data_collator, output_dir, kwargs)
    
    # 6. Evaluation Stage
    eval_metrics = execute_evaluation(trainer, output_dir, model, tokenizer)
    
    return eval_metrics

# ----------------------------
# DAG definition
# ----------------------------
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="train_multi_v1",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'sentiment', 'training']
) as dag:

    # Pass the config dictionary to the orchestrated main function
    training_kwargs = {
        "data_path": "/opt/airflow/data/mix3.csv", 
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "output_dir": "/opt/airflow/models/sentiment_model_v1",
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01
    }

    train_task = PythonOperator(
        task_id="train_multi_task",
        python_callable=main_pipeline,
        op_kwargs=training_kwargs
    )
