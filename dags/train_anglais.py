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


def train_model(**kwargs):
    """
    Main training pipeline to fine-tune the sentiment model.
    Parameters are passable as arguments, making it Airflow-compatible.
    """
    # 1. Retrieve config parameters from kwargs (passed via PythonOperator)
    data_path = kwargs.get("data_path", "/opt/airflow/data/twitter_training.csv")
    model_name = kwargs.get("model_name", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
    output_dir = kwargs.get("output_dir", "/opt/airflow/models/sentiment_model")
    num_train_epochs = kwargs.get("num_train_epochs", 3.0)
    per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 16)
    per_device_eval_batch_size = kwargs.get("per_device_eval_batch_size", 16)
    learning_rate = kwargs.get("learning_rate", 2e-5)
    evaluation_strategy = kwargs.get("evaluation_strategy", "epoch")
    save_strategy = kwargs.get("save_strategy", "epoch")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Data Loading
    print(f"Loading data from {data_path}...")
    col_names = ["id", "game_title", "sentiment_label", "tweet_text"]
    df = pd.read_csv(data_path, header=None, names=col_names)
    
    # 3. Data Cleaning
    # Drop nulls in the tweet text column
    df = df.dropna(subset=['tweet_text'])
    # Drop rows where sentiment_label is null
    df = df.dropna(subset=['sentiment_label'])
    
    # Strip whitespace from labels and standardize case
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.strip().str.capitalize()
    
    # 4. Label Encoding
    # Map the 4 string labels to integers
    label_map = {
        "Positive": 0,
        "Negative": 1,
        "Neutral": 2,
        "Irrelevant": 3
    }
    
    # Filter to ensure only valid labels are present
    df = df[df['sentiment_label'].isin(label_map.keys())]
    df['label'] = df['sentiment_label'].map(label_map)
    
    # Keep necessary columns
    df = df[['tweet_text', 'label']]
    df['tweet_text'] = df['tweet_text'].astype(str)
    
    # 5. Train/Validation Split (80/20)
    print("Splitting data into train and validation sets (80/20)...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # 6. Tokenization
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['tweet_text'], truncation=True, max_length=128)
        
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 7. Model Setup
    print(f"Loading AutoModelForSequenceClassification for {model_name} with num_labels=4...")
    id2label = {v: k for k, v in label_map.items()}
    
    # Transformers will use CUDA automatically if available
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        id2label=id2label,
        label2id=label_map,
        ignore_mismatched_sizes=True
    )
    
    # 8. TrainingArguments Configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,             
        per_device_train_batch_size=per_device_train_batch_size,   
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        learning_rate=learning_rate,               
        load_best_model_at_end=True,      
        metric_for_best_model="f1_weighted",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        # Default environment will automatically use CUDA if available via torch backend
    )
    
    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,     
    )
    
    # 10. Start Training
    print("Starting Model Training...")
    trainer.train()
    
    # 11. Evaluate After Training
    print("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    
    print("------------------------")
    print("Final Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("------------------------")
        
    # 12. Save final model and tokenizer
    print(f"Saving fine-tuned model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return eval_metrics

# ----------------------------
# DAG definition
# ----------------------------
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="fine_tune_social_sentiment",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'sentiment', 'training']
) as dag:

    # Define training config to be passed to the training function
    training_kwargs = {
        "data_path": "/opt/airflow/data/twitter_training.csv", 
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "output_dir": "/opt/airflow/models/twitter_sentiment_finetuned",
        "num_train_epochs": 3.0,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch"
    }

    train_task = PythonOperator(
        task_id="train_sentiment_model_task",
        python_callable=train_model,
        op_kwargs=training_kwargs
    )
