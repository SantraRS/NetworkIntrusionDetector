import os

# Define the absolute path to the project's root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Data Folders ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# --- Data File Paths ---
# Interim files
TRAIN_DF_PATH = os.path.join(INTERIM_DATA_DIR, "train_df.csv")
TEST_DF_PATH = os.path.join(INTERIM_DATA_DIR, "test_df.csv")

# Processed files (as numpy arrays for fast loading)
X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train_scaled.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test_scaled.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")


# --- Model Folders ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_agent.zip")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# --- Media/Reports Folders ---
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")
PLOT_DIR = os.path.join(MEDIA_DIR, "plots")
CONFUSION_MATRIX_PATH = os.path.join(PLOT_DIR, "confusion_matrix.png")

def create_dirs():
    """
    Creates all necessary directories for the project.
    """
    for path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, PLOT_DIR]:
        os.makedirs(path, exist_ok=True)
    print("All project directories created.")

if __name__ == "__main__":
    create_dirs()