from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil
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
import logging

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

# --- File Paths ---
PSEUDO_DATA_PATH = "/opt/airflow/data/tunisia_plus_pseudo.csv"
MERGED_DATA_PATH = "/opt/airflow/data/merged_data.csv"
TRAIN_DATA_PATH = "/opt/airflow/data/train_pseudo.parquet"
TEST_DATA_PATH = "/opt/airflow/data/test_pseudo.parquet"

# Load from the fine-tuned model (the one used for pseudo labeling)
MODEL_NAME = "/opt/airflow/models/pretrained_model"
BASE_MODEL_DIR = "/opt/airflow/models/base_model_pseudo"
TEMP_TRAIN_DIR = "/opt/airflow/models/temp_training_pseudo"
FINAL_MODEL_DIR = "/opt/airflow/models/pseudo_finetuned_model/"

def preprocess_pseudo(df):
    """
    Standardize the incoming pseudo-labeled data.
    """
    # Rename predicted_label to label if necessary
    if 'predicted_label' in df.columns:
        df = df.rename(columns={'predicted_label': 'label'})
        
    if 'text' not in df.columns or 'label' not in df.columns:
        return pd.DataFrame()
        
    df = df[['text', 'label']].copy()
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    # Map back to IDs based on usual model format
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df = df[df['label'].isin(mapping.keys())]
    df['label'] = df['label'].map(mapping)
    return df

def load_data(**kwargs):
    """
    1. Load Pseudo-labeled Dataset mixed with random samples from Merged Data
    """
    logging.info(f"Loading Pseudo-labeled data from {PSEUDO_DATA_PATH}")
    df_pseudo = pd.read_csv(PSEUDO_DATA_PATH)
    
   df_pseudo = preprocess_pseudo(df_pseudo)
    
    logging.info(f"Loading Original Merged data from {MERGED_DATA_PATH}")
    if os.path.exists(MERGED_DATA_PATH):
        df_merged = pd.read_csv(MERGED_DATA_PATH)
        df_merged = preprocess_pseudo(df_merged) # reuse the same cleaning/mapping
        
        # Sample 8000 rows to prevent catastrophic forgetting
        n_samples = min(8000, len(df_merged))
        logging.info(f"Sampling {n_samples} from the merged original dataset.")
        df_merged_subset = df_merged.sample(n=n_samples, random_state=42)
        
        df = pd.concat([df_pseudo, df_merged_subset], ignore_index=True)
    else:
        logging.warning("Merged dataset not found! Training on pseudo-labels only.")
        df = df_pseudo
    
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split dataset into train/test (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logging.info(f"Train subset size from pseudo labels: {len(train_df)}")
    logging.info(f"Test subset size from pseudo labels: {len(test_df)}")
    
    # Save intermediate files for train_model to pick up
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)
    logging.info("Saved train and test datasets.")

def load_model(**kwargs):
    """
    2. Load Base Model
    """
    logging.info(f"Loading already fine-tuned model from {MODEL_NAME}")
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    
    # Load model 
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
        return tokenizer(examples['text'], padding=False, truncation=True, max_length=128)
        
    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Padding applied dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_DIR, num_labels=3)
    
    # Prevent catastrophic forgetting: Freeze embedding layer
    if hasattr(model, 'roberta'):
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        logging.info("Freezing RoBERTa embedding layer.")
    
    # TrainingArguments config - low learning rate for pure pseudo labels
    training_args = TrainingArguments(
        output_dir=TEMP_TRAIN_DIR,
        num_train_epochs=2,             
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=False,
        learning_rate=5e-6,               # Very low rate
        evaluation_strategy="epoch",      
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        report_to="none",
        disable_tqdm=True,
        logging_steps=10
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,     
    )
    
    logging.info("Starting Incremental Fine-Tuning on Pseudo Labels...")
    trainer.train()
    
    logging.info("Evaluating Model...")
    eval_metrics = trainer.evaluate()
    
    logging.info("Final Evaluation Metrics:")
    for key, value in eval_metrics.items():
        logging.info(f"  {key}: {value}")
        
    trainer.save_model(TEMP_TRAIN_DIR)
    tokenizer.save_pretrained(TEMP_TRAIN_DIR)

def save_model(**kwargs):
    """
    7. Save Model
    """
    logging.info(f"Saving final pseudo-finetuned model to {FINAL_MODEL_DIR}")
    
    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
        
    shutil.copytree(TEMP_TRAIN_DIR, FINAL_MODEL_DIR)
    logging.info("Model saved successfully.")

# --- DAG Definition ---
with DAG(
    dag_id="pseudo_fine_tuning_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'sentiment', 'trainer', 'pseudo-labels', 'finetuning']
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

    task_load_data >> task_load_model >> task_train_model >> task_save_model
