"""
Multilingual Sentiment Dataset Builder
Expands an English sentiment dataset into a multi-lingual one (English, French, Arabic).
"""

import os
import pandas as pd
import numpy as np
import re
import html
import torch
from transformers import pipeline


# Airflow specific imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def load_data(input_file: str) -> pd.DataFrame:
    """1. Load CSV file and keep only 'text' and 'label' columns, dropping missing values."""
    print("----------------------------------")
    print(f"1. LOADING DATA: {input_file}")
    
    # If the file hasn't been created yet or path is different in your environment, adjust this path.
    # To be extremely robust, let's load it and resolve column naming dynamically.
    try:
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return pd.DataFrame()
        
    # Standardize column mappings if they vary
    if 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    if 'tweet_text' in df.columns:
        df = df.rename(columns={'tweet_text': 'text'})
        
    # If headers are missing, assume order mapping fallback
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Required columns missing, re-reading with fallback structure...")
        df = pd.read_csv(input_file, header=None, names=['textID', 'text', 'selected_text', 'label'])
        
    # Select strictly 'text' and 'label'
    df = df[['text', 'label']].copy()
    
    # Clean up empty formats and drop rows containing pure NaNs
    df = df.dropna(subset=['text', 'label'])
    
    # Ensure they are lowercase for labels
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    print(f"Loaded {len(df)} non-null rows.")
    return df

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """2. Cleans out duplicates, short lengths, and noise (like URLs only)."""
    print("----------------------------------")
    print("2. FILTERING DATA")
    initial_len = len(df)
    print(f"Size before filtering: {initial_len} rows")
    
    # 1. Cast 'text' safely and strip wrapping white spaces
    df['text'] = df['text'].astype(str).str.strip()
    
    # 2. Drop duplicates on the "text" column
    df = df.drop_duplicates(subset=['text'])
    
    # 3. Remove specifically short texts (< 5 chars)
    df = df[df['text'].str.len() >= 5]
    
    # 4. Remove rows that are ONLY URLs, using a simple regex check
    # Matches strings that consist of one or more URLs and optional spaces only
    url_only_idx = df['text'].str.match(r'^(https?://\S+\s*)+$')
    df = df[~url_only_idx]
    
    final_len = len(df)
    print(f"Size after filtering: {final_len} rows (Removed {initial_len - final_len} irrelevant/noisy rows)")
    return df

def split_for_translation(df: pd.DataFrame):
    """3. Samples roughly 33% for FR translation, 33% for AR translation, leaving 34% original."""
    print("----------------------------------")
    print("3. SAMPLING FOR TRANSLATION")
    
    # Shuffle first with fixed random state for reproducibility
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    total_len = len(df_shuffled)
    count_fr = int(0.33 * total_len)
    count_ar = int(0.33 * total_len)
    
    # Slice the dataframe into the respective percentiles
    df_fr = df_shuffled.iloc[:count_fr].copy()
    df_ar = df_shuffled.iloc[count_fr:count_fr+count_ar].copy()
    df_en = df_shuffled.iloc[count_fr+count_ar:].copy()
    
    print(f"Split sizes -> English (34%): {len(df_en)}, French (33%): {len(df_fr)}, Arabic (33%): {len(df_ar)}")
    return df_en, df_fr, df_ar

def clean_translated_text(text: str) -> str:
    """5. Cleans HTML entities, strips extra spaces and secures UTF-8 encoding compliance."""
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (&amp;, &lt;, etc.)
    text = html.unescape(text)
    # Strip multiple consecutive spaces natively
    text = re.sub(r'\s+', ' ', text)
    # Ensure utf-8 safety
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text.strip()

def translate_texts(df: pd.DataFrame, target_lang_model: str, batch_size: int = 32) -> pd.DataFrame:
    """4. Configures a batch pipeline translating chunks of the target dataset with automatic GPU support."""
    print("----------------------------------")
    print(f"4. TRANSLATION (Model: {target_lang_model})")
    
    if df.empty:
        return df

    # Automatically identify if a GPU is available to allocate computation mappings
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device {'GPU (cuda:0)' if device == 0 else 'CPU'} with batch_size: {batch_size}")
    
    # Initialize translation pipeline via huggingface transformers
    print("Loading model weights (this might take a minute on first run)...")
    translator = pipeline("translation", model=target_lang_model, device=device)
    
    texts = df['text'].tolist()
    translations = []
    
    # Use tqdm to view progression status dynamically 
    print(f"Translating {len(texts)} entries...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Process"):
        batch_text = texts[i:i+batch_size]
        try:
            # Truncation applies safely to drop tokens beyond the model's acceptable input boundary bounds 
            results = translator(batch_text, truncation=True)
            batch_translations = [res['translation_text'] for res in results]
            translations.extend(batch_translations)
        except Exception as e:
            # Graceful Fallback Design Architecture
            # If translation errors (Memory/Out Of Bounds/etc.), we append original English sequences
            print(f"Error in batch starting at {i}: {e}. Falling back to original english text strings.")
            translations.extend(batch_text)
            
    # Apply cleanup sequence
    print("5. CLEANING TRANSLATED TEXT (HTML entities formatting, space stripping)...")
    cleaned_translations = [clean_translated_text(t) for t in translations]
    
    df['text'] = cleaned_translations
    
    # Memory flushing to protect resources & reduce VRAM impact between loads
    del translator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return df

def merge_and_export(df_en: pd.DataFrame, df_fr: pd.DataFrame, df_ar: pd.DataFrame, output_path: str):
    """6. Merges data sources concurrently and shaves down final export files."""
    print("----------------------------------")
    print("6. MERGING DATA")
    
    # Concat splits logically 
    merged_df = pd.concat([df_en, df_fr, df_ar], ignore_index=True)
    
    # Final random state shuffle before finalizing output dependencies
    merged_df = merged_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("----------------------------------")
    print("7. FINAL OUTPUT")
    print(f"Saving finalized generated structure to: {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    print(f"Final dataset size: {len(merged_df)} rows")
    print("\nClass Distribution:")
    print(merged_df['label'].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    
    print("\nSample rows generated:")
    print(merged_df.head())

def build_pipeline():
    """Execution Orchestrator."""
    input_file = "/opt/airflow/data/twitter_training.csv"
    output_file = "/opt/airflow/data/multilingual_dataset.csv"
    
    # Configurable variables
    batch_size = 32
    model_fr = "Helsinki-NLP/opus-mt-en-fr"
    model_ar = "Helsinki-NLP/opus-mt-en-ar"

    # Pipeline Launching
    df = load_data(input_file)
    if df.empty:
        return
        
    df = filter_data(df)
    df_en, df_fr, df_ar = split_for_translation(df)
    
    print("\n>>> Translating French Dataset Sector")
    df_fr = translate_texts(df_fr, target_lang_model=model_fr, batch_size=batch_size)
    
    print("\n>>> Translating Arabic Dataset Sector")
    df_ar = translate_texts(df_ar, target_lang_model=model_ar, batch_size=batch_size)
    
    merge_and_export(df_en, df_fr, df_ar, output_file)
    print("\nPipeline Completion! ✅")

import warnings
warnings.filterwarnings("ignore")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id="build_multilingual_dataset",
    default_args=default_args,
    description="Pipeline to translate and construct a multilingual sentiment dataset",
    schedule_interval=None,
    catchup=False,
    tags=['nlp', 'data-prep'],
) as dag:

    # Wrap the entire orchestrator in a single PythonOperator to bypass XCom payload limits 
    # and execute translations natively on the worker node constraint-free.
    build_dataset_task = PythonOperator(
        task_id="translate_and_merge_data",
        python_callable=build_pipeline,
    )
