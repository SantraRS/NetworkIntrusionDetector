import pandas as pd
import numpy as np
import glob
import os
import kagglehub
from sklearn.model_selection import train_test_split
from src.utils import helpers # Import path constants
def download_data():
    """
    Downloads the dataset from KaggleHub to the local cache.
    (From Cell 1)
    """
    print("Downloading dataset...")
    # Remove the 'path' argument. This will download to the default
    # kagglehub cache and return the path to that cache directory.
    path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
    
    print(f"Dataset downloaded to cache: {path}")
    return path

# ... rest of the script ...

def load_clean_merge_data(raw_path):
    """
    Loads all CSVs from raw path, cleans, and merges them.
    (From Cells 5 & 7)
    """
    print("Loading, cleaning, and merging data...")
    all_files = glob.glob(os.path.join(raw_path, "*.csv"))
    
    if not all_files:
        print(f"No CSV files found in {raw_path}. Make sure download was successful.")
        return None

    df_list = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Clean column names (Cell 7)
    combined_df.columns = combined_df.columns.str.strip()
    
    # Handle infinite values and drop NaNs (Cell 7)
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)
    
    print(f"Data merged and cleaned. Shape: {combined_df.shape}")
    return combined_df

def balance_and_split_data(df):
    """
    Balances the dataset and performs train/test split and sampling.
    (From Cells 9 & 11)
    """
    print("Balancing and splitting data...")
    
    # Balance dataset (Cell 9)
    attack_df = df[df['Label'] != 'BENIGN']
    benign_df = df[df['Label'] == 'BENIGN'].sample(n=len(attack_df), random_state=42)
    hackathon_df = pd.concat([benign_df, attack_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split train/test (Cell 11)
    train_df, test_df = train_test_split(hackathon_df, test_size=0.2, random_state=42)
    
    # Balance training sample (Cell 11)
    attack_train = train_df[train_df['Label'] != 'BENIGN']
    benign_train = train_df[train_df['Label'] == 'BENIGN']
    n_samples = min(20000, len(attack_train), len(benign_train))
    
    attack_sample = attack_train.sample(n=n_samples, random_state=42)
    benign_sample = benign_train.sample(n=n_samples, random_state=42)
    train_df = pd.concat([attack_sample, benign_sample]).sample(frac=1, random_state=42)
    
    # Sample 20k from test set (Cell 11)
    test_df = test_df.sample(n=min(20000, len(test_df)), random_state=42)
    
    print(f"Data processing complete.")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    return train_df, test_df

def main():
    # Ensure all directories exist
    helpers.create_dirs()
    
    # Step 1: Download data
    raw_path = download_data()
    
    # Step 2: Load, clean, and merge
    combined_df = load_clean_merge_data(raw_path)
    
    if combined_df is not None:
        # Step 3: Balance, split, and sample
        train_df, test_df = balance_and_split_data(combined_df)
        
        # Step 4: Save interim data
        train_df.to_csv(helpers.TRAIN_DF_PATH, index=False)
        test_df.to_csv(helpers.TEST_DF_PATH, index=False)
        print(f"Interim train and test sets saved to {helpers.INTERIM_DATA_DIR}")

if __name__ == "__main__":
    main()