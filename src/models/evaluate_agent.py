import numpy as np
from stable_baselines3 import DQN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import helpers
import os

def evaluate_agent():
    """
    Loads the trained model and test data, then runs evaluation and saves reports.
    (From Cell 23)
    """
    print("Loading trained model and test data...")
    try:
        model = DQN.load(helpers.MODEL_PATH)
        X_test_scaled = np.load(helpers.X_TEST_PATH)
        y_test = np.load(helpers.Y_TEST_PATH)
    except FileNotFoundError:
        print("Error: Model or test data not found.")
        print("Please run 'src/features/build_features.py' and 'src/models/train_agent.py' first.")
        return

    print(f"Making predictions on {len(y_test)} test samples...")
    # Note: We don't need to load the env, just the data.
    # deterministic=True gives the best-known action, not a sampled one.
    predictions, _ = model.predict(X_test_scaled, deterministic=True)

    # --- Performance Report ---
    
    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n--- Model Performance Report ---")
    print(f"Model Accuracy: {accuracy:.2%}\n")

    # Classification Report
    print("Classification Report:")
    report = classification_report(y_test, predictions, target_names=['Benign (0)', 'Attack (1)'], labels=[0, 1])
    print(report)

    # Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.savefig(helpers.CONFUSION_MATRIX_PATH)
    print(f"Confusion Matrix saved to {helpers.CONFUSION_MATRIX_PATH}")
    # plt.show() # Uncomment if you want it to pop up when running interactively

if __name__ == "__main__":
    helpers.create_dirs() # Ensure directories exist
    evaluate_agent()