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
DATA_PATH = "/opt/airflow/data/merged_data.csv"
TRAIN_DATA_PATH = "/opt/airflow/data/train_additive.parquet"
TEST_DATA_PATH = "/opt/airflow/data/test_additive.parquet"

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BASE_MODEL_DIR = "/opt/airflow/models/base_model_additive"
TEMP_TRAIN_DIR = "/opt/airflow/models/temp_training_additive"
FINAL_MODEL_DIR = "/opt/airflow/models/additive_finetuned_model_real/"

def load_data(**kwargs):
    logging.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    df = df[['text', 'label']]
    
    label_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df = df[df['label'].isin(label_mapping.keys())]
    df['label'] = df['label'].map(label_mapping)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logging.info(f"Train set size: {len(train_df)}")
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)

def load_model(**kwargs):
    logging.info(f"Downloading/Loading base model from {MODEL_NAME}")
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.save_pretrained(BASE_MODEL_DIR)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1_weighted': f1}

def train_model(**kwargs):
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=False, truncation=True, max_length=128)
        
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_DIR, num_labels=3)
    
    # ----------------------------------------------------
    # ADDITIVE FINE-TUNING STRATEGY (LoRA)
    # ----------------------------------------------------
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Configure LoRA
        # r: rank of the update matrices
        # lora_alpha: LoRA scaling factor
        # target_modules: the projection matrices to apply LoRA on
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=8, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=["query", "value"]  # XLM-RoBERTa uses query and value inside self-attention
        )
        
        # This function freezes the main model weights and adds trainable LoRA adapters
        model = get_peft_model(model, lora_config)
        
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logging.info(f"Additive Fine-tuning (LoRA) Applied! Trainable parameters: {trainable_params} / {all_param}")
        
    except ImportError:
        logging.warning("PEFT library not found. Falling back to freezing embeddings (partial) instead.")
        if hasattr(model, 'roberta'):
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir=TEMP_TRAIN_DIR,
        num_train_epochs=3,             
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=False,
        learning_rate=1e-4,               # LoRA allows for slightly higher learning rates
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
    
    logging.info("Starting Additive Fine-Tuning...")
    trainer.train()
    
    # Evaluate and Save
    eval_metrics = trainer.evaluate()
    for key, value in eval_metrics.items():
        logging.info(f"  {key}: {value}")
        
    trainer.save_model(TEMP_TRAIN_DIR)
    tokenizer.save_pretrained(TEMP_TRAIN_DIR)

def save_model(**kwargs):
    logging.info(f"Saving additive finetuned model to {FINAL_MODEL_DIR}")
    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
    shutil.copytree(TEMP_TRAIN_DIR, FINAL_MODEL_DIR)

with DAG(
    dag_id="additive_finetuning_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'additive-finetune', 'lora', 'peft']
) as dag:
    
    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="load_model", python_callable=load_model)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model)
    t4 = PythonOperator(task_id="save_model", python_callable=save_model)

    t1 >> t2 >> t3 >> t4
