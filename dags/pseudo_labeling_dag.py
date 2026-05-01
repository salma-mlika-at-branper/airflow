"""
DAG for pseudo-labeling the tunisia_plus dataset using a fine-tuned sentiment model.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Ensure string return
        text = self.texts[idx]
        return str(text) if pd.notna(text) else ""


def perform_pseudo_labeling(**kwargs):
    """
    PythonOperator callable that performs pseudo-labeling inference 
    using parameters passed via op_kwargs.
    """
    model_dir = kwargs.get("model_dir", "/opt/airflow/models/pretrained_model")
    input_csv = kwargs.get("input_csv", "/opt/airflow/data/tunisia_plus.csv")
    output_csv = kwargs.get("output_csv", "/opt/airflow/data/tunisia_plus_pseudo.csv")
    batch_size = kwargs.get("batch_size", 64)
    confidence_threshold = kwargs.get("confidence_threshold", 0.90)


    print(f"Loading data from {input_csv}...")
    # Read linearly without splitting on commas, as the file lacks headers and contains interior commas
    with open(input_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame({"text": lines})

    original_input_count = len(df)
    already_labeled_texts = set()
    
    if os.path.exists(output_csv):
        print(f"Found existing output file {output_csv}, loading to filter out already labeled samples...")
        existing_df = pd.read_csv(output_csv)
        if 'text' in existing_df.columns:
            already_labeled_texts = set(existing_df['text'].astype(str).str.strip())
            
    # Filter out already labeled samples
    df['text_clean'] = df['text'].astype(str).str.strip()
    df = df[~df['text_clean'].isin(already_labeled_texts)]
    df = df.drop(columns=['text_clean'])
    
    new_samples_count = len(df)
    skipped_count = original_input_count - new_samples_count
    
    print(f"Found {skipped_count} samples already labeled. Processing {new_samples_count} new samples.")
    
    if new_samples_count == 0:
        print("No new samples to process. Exiting.")
        return
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model and tokenizer from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Label mapping (id2label)
    id2label = model.config.id2label
    if not id2label:
        print("Warning: id2label not found in model config. Using default indices as labels.")
        id2label = {i: f"LABEL_{i}" for i in range(model.config.num_labels)}
    
    print(f"Label mapping found: {id2label}")
    
    # DataLoader preparation
    dataset = TextDataset(df["text"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predicted_labels = []
    all_confidence_scores = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_texts in dataloader:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Move tokens to device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Compute probabilities using Softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the maximum probability and the corresponding class index
            confidence_scores, predicted_indices = torch.max(probs, dim=-1)
            
            # Collect results
            all_confidence_scores.extend(confidence_scores.cpu().numpy())
            all_predicted_labels.extend([id2label[idx.item()] for idx in predicted_indices])
    
    # Assign predictions back to the original dataframe
    df["predicted_label"] = all_predicted_labels
    df["confidence_score"] = all_confidence_scores
    
    # Filter out predictions that are below the confidence threshold
    original_count = len(df)
    filtered_df = df[df["confidence_score"] >= confidence_threshold]
    kept_count = len(filtered_df)
    
    print("\n--- Pseudo-Labeling Summary ---")
    print(f"Already labeled samples skipped: {skipped_count}")
    print(f"Total new samples processed: {original_count}")
    print(f"New samples kept (confidence >= {confidence_threshold}): {kept_count}")
    print(f"New samples discarded: {original_count - kept_count}")
    
    if kept_count > 0:
        print("\nDistribution of predicted labels in confident subset:")
        label_counts = filtered_df["predicted_label"].value_counts()
        for label, count in label_counts.items():
            percentage = (count / kept_count) * 100
            print(f"  - {label}: {count} ({percentage:.2f}%)")
    else:
        print("\nNo samples met the confidence threshold.")
        
    # Keep only requested columns
    output_columns = ["text", "predicted_label", "confidence_score"]
    output_df = filtered_df[output_columns]
    
    # Output to CSV
    output_dir_path = os.path.dirname(output_csv)
    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
        
    print(f"\nAppending new pseudo-labeled dataset to '{output_csv}'...")
    output_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
    print("Done!")
# ----------------------------
# DAG definition
# ----------------------------

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="pseudo_labeling_dag",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=['nlp', 'sentiment', 'inference']
) as dag:

    # Paths align with standard Airflow docker mounts from earlier DAGs
    # Update 'model_dir' if your trained model is under a different path
    # e.g., "/opt/airflow/models/sentiment_model_v1"
    inference_kwargs = {
        "model_dir": "/opt/airflow/models/pretrained_model", 
        "input_csv": "/opt/airflow/data/tunisia_plus.csv",
        "output_csv": "/opt/airflow/data/tunisia_plus_pseudo.csv",
        "batch_size": 64,
        "confidence_threshold": 0.90
    }

    pseudo_label_task = PythonOperator(
        task_id="run_pseudo_labeling",
        python_callable=perform_pseudo_labeling,
        op_kwargs=inference_kwargs
    )
