import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.utils import helpers # Import path constants

def create_and_scale_features():
    """
    Loads interim data, creates features/labels, scales, and saves processed data.
    (From Cells 13 & 15)
    """
    print("Loading interim data...")
    try:
        train_df = pd.read_csv(helpers.TRAIN_DF_PATH)
        test_df = pd.read_csv(helpers.TEST_DF_PATH)
    except FileNotFoundError:
        print(f"Error: Interim data not found. Please run 'src/data/make_dataset.py' first.")
        return

    print("Creating features (X) and labels (y)...")
    # Features & labels (Cell 13)
    X_train = train_df.drop(columns=['Label'])
    y_train = (train_df['Label'] != 'BENIGN').astype(int).values
    
    X_test = test_df.drop(columns=['Label'])
    y_test = (test_df['Label'] != 'BENIGN').astype(int).values

    print("Scaling data...")
    # Scaling (Cell 15)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Save the scaler (From Cell 20)
    joblib.dump(scaler, helpers.SCALER_PATH)
    print(f"Scaler saved to {helpers.SCALER_PATH}")
    
    # Save processed data as numpy arrays
    np.save(helpers.X_TRAIN_PATH, X_train_scaled)
    np.save(helpers.Y_TRAIN_PATH, y_train)
    np.save(helpers.X_TEST_PATH, X_test_scaled)
    np.save(helpers.Y_TEST_PATH, y_test)
    
    print(f"Processed data saved to {helpers.PROCESSED_DATA_DIR}")
    print("\nTrain class distribution:\n", pd.Series(y_train).value_counts())
    print("\nTest class distribution:\n", pd.Series(y_test).value_counts())

if __name__ == "__main__":
    helpers.create_dirs() # Ensure directories exist
    create_and_scale_features()