import os
import logging
import pandas as pd
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# Define pipeline paths
DATA_DIR = "/opt/airflow/data"
MULTI_DATA_PATH = os.path.join(DATA_DIR, "multilingual_dataset.csv")
TUNISIAN_DATA_PATH = os.path.join(DATA_DIR, "tunisainone.csv")
DERJA_DATA_PATH = os.path.join(DATA_DIR, "derja_arbi.csv")
MERGED_OUTPUT_PATH = os.path.join(DATA_DIR, "merged_data.csv")

# Temporary files for passing data between tasks
TEMP_MULTI = os.path.join(DATA_DIR, "temp_multi.parquet")
TEMP_TUNISIAN = os.path.join(DATA_DIR, "temp_tunisain.parquet")
TEMP_DERJA = os.path.join(DATA_DIR, "temp_derja.parquet")
TEMP_COMBINED = os.path.join(DATA_DIR, "temp_combined.parquet")
TEMP_CLEANED = os.path.join(DATA_DIR, "temp_cleaned.parquet")

def load_data(**kwargs):
    """
    Task to safely load datasets and handle missing files.
    """
    logging.info(f"Looking for data in {DATA_DIR}")
    
    # Load multilingual dataset
    if os.path.exists(MULTI_DATA_PATH):
        try:
            df_multi = pd.read_csv(MULTI_DATA_PATH, on_bad_lines='skip')
            logging.info(f"Successfully loaded {MULTI_DATA_PATH}. Shape: {df_multi.shape}")
            df_multi.to_parquet(TEMP_MULTI)
        except Exception as e:
            logging.error(f"Error loading {MULTI_DATA_PATH}: {e}")
            pd.DataFrame(columns=['text', 'label']).to_parquet(TEMP_MULTI)
    else:
        logging.warning(f"File not found: {MULTI_DATA_PATH}. Proceeding with empty dataframe.")
        pd.DataFrame(columns=['text', 'label']).to_parquet(TEMP_MULTI)

    # Load tunisain dataset
    if os.path.exists(TUNISIAN_DATA_PATH):
        try:
            df_tun = pd.read_csv(TUNISIAN_DATA_PATH, on_bad_lines='skip')
            logging.info(f"Successfully loaded {TUNISIAN_DATA_PATH}. Shape: {df_tun.shape}")
            df_tun.to_parquet(TEMP_TUNISIAN)
        except Exception as e:
            logging.error(f"Error loading {TUNISIAN_DATA_PATH}: {e}")
            pd.DataFrame(columns=['text', 'label']).to_parquet(TEMP_TUNISIAN)
    else:
        logging.warning(f"File not found: {TUNISIAN_DATA_PATH}. Proceeding with empty dataframe.")
        pd.DataFrame(columns=['text', 'label']).to_parquet(TEMP_TUNISIAN)
    # Load derja dataset
    if os.path.exists(DERJA_DATA_PATH):
        try:
            df_derja = pd.read_csv(DERJA_DATA_PATH, on_bad_lines='skip')
            logging.info(f"Successfully loaded {DERJA_DATA_PATH}. Shape: {df_derja.shape}")
            df_derja.to_parquet(TEMP_DERJA)
        except Exception as e:
            logging.error(f"Error loading {DERJA_DATA_PATH}: {e}")
            pd.DataFrame(columns=['text', 'label']).to_parquet(TEMP_DERJA)
    else:
        logging.warning(f"File not found: {DERJA_DATA_PATH}. Proceeding with empty dataframe.")
        pd.DataFrame(columns=['text', 'label']).to_parquet(TEMP_DERJA)
    

def combine_data(**kwargs):
    """
    Task to standardize columns from both datasets and merge them into one.
    """
    logging.info("Starting combine_data task.")
    
    df_multi = pd.read_parquet(TEMP_MULTI)
    df_tun = pd.read_parquet(TEMP_TUNISIAN)
    df_derja = pd.read_parquet(TEMP_DERJA)
    
    def standardize_columns(df, dataset_name):
        if df.empty:
            logging.info(f"{dataset_name} is empty. Skipping standardization.")
            return pd.DataFrame(columns=['text', 'label'])
            
        # Ensure at least 2 columns
        if len(df.columns) < 2:
            return pd.DataFrame(columns=['text', 'label'])
            
        # Extract the first two columns, regardless of their names initially
        df = df.iloc[:, :2].copy()
        df.columns = ['col0', 'col1']
        
        valid_labels = ['positive', 'negative', 'neutral']
        
        # Sometimes the 'text' column gets read as 'label' and 'label' as 'text'
        # We identify which column holds the known label values
        mask_col0_is_label = df['col0'].astype(str).str.lower().str.strip().isin(valid_labels)
        
        text_series = df['col0'].copy()
        label_series = df['col1'].copy()
        
        # Swap places wherever col0 was identified as holding the label
        text_series.loc[mask_col0_is_label] = df.loc[mask_col0_is_label, 'col1']
        label_series.loc[mask_col0_is_label] = df.loc[mask_col0_is_label, 'col0']
        
        std_df = pd.DataFrame({
            'text': text_series,
            'label': label_series
        })
        logging.info(f"Standardized {dataset_name}. Resulting shape: {std_df.shape}")
        return std_df

    df_multi_std = standardize_columns(df_multi, 'Multilingual Dataset')
    df_tun_std = standardize_columns(df_tun, 'Tunisian Dataset')
    df_derja_std = standardize_columns(df_derja, 'Derja Dataset')
    # Merge both datasets
    combined_df = pd.concat([df_multi_std, df_tun_std, df_derja_std], ignore_index=True)
    logging.info(f"Combined data shape: {combined_df.shape}")
    
    # Save to intermediate parquet
    combined_df.to_parquet(TEMP_COMBINED)


def clean_data(**kwargs):
    """
    Task to clean text, enforce rules, and handle dataset normalization.
    """
    logging.info("Starting clean_data task.")
    df = pd.read_parquet(TEMP_COMBINED)
    
    initial_rows = len(df)
    
    # 1. Remove null values
    df.dropna(subset=['text', 'label'], inplace=True)
    
    # 2. Convert text to string
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    
    # 3. Normalize text (lowercase, strip spaces)
    df['text'] = df['text'].str.lower().str.strip()
    df['label'] = df['label'].str.lower().str.strip()
    
    # 4. Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # 5. Remove very short texts (< 5 characters)
    df = df[df['text'].str.len() >= 5]
    
    # 6. Standardize labels to allowed categories only
    valid_labels = ['positive', 'negative', 'neutral']
    df = df[df['label'].isin(valid_labels)]
    
    final_rows = len(df)
    logging.info(f"Cleaning complete. Rows went from {initial_rows} -> {final_rows}")
    
    df.to_parquet(TEMP_CLEANED)


def save_dataset(**kwargs):
    """
    Task to shuffle the clean dataset and save it to the final output file.
    """
    logging.info(f"Starting save_dataset task.")
    df = pd.read_parquet(TEMP_CLEANED)
    
    # Shuffle rows randomly with fixed seed (42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info("Shuffled dataset with seed 42.")
    
    # Save final dataset
    df.to_csv(MERGED_OUTPUT_PATH, index=False)
    logging.info(f"Saved merged and cleaned dataset to {MERGED_OUTPUT_PATH}")
    logging.info(f"Final Output shape: {df.shape}")
    
    # Optional cleanup of intermediate files to save disk space
    temp_files = [ TEMP_MULTI,TEMP_TUNISIAN, TEMP_DERJA,TEMP_COMBINED, TEMP_CLEANED]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logging.info(f"Removed temp file {temp_file}")


# Define basic args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Instantiate the DAG
with DAG(
    dag_id='data_preparation_pipeline',
    default_args=default_args,
    description='ETL pipeline for sentiment analysis datasets',
    schedule_interval=None,  # Run manually
    catchup=False,
    tags=['etl', 'sentiment_analysis', 'data_prep']
) as dag:

    # Define tasks
    t_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    t_combine = PythonOperator(
        task_id='combine_data',
        python_callable=combine_data,
    )

    t_clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )

    t_save = PythonOperator(
        task_id='save_dataset',
        python_callable=save_dataset,
    )

    # Set dependencies
    t_load >> t_combine >> t_clean >> t_save
