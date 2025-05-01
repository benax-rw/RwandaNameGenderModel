import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)

# Define paths
model_path = "model/logistic_model.joblib"
vectorizer_path = "model/vectorizer.joblib"
log_dir = "logs"
metrics_log_path = os.path.join(log_dir, "metrics_log.txt")

# Create directories if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# Step 1: Load and preprocess dataset
file_path = "dataset/rwandan_names.csv"
logging.info("Loading dataset from file...")
data = pd.read_csv(file_path)
logging.info("Dataset loaded. Checking for missing values...")

# Drop rows with NaN values in 'name' or 'gender' columns
data = data.dropna(subset=['name', 'gender'])
logging.info(f"Dataset loaded with {len(data)} records after dropping NaNs.")

# Step 2: Split dataset into training, validation, and test sets
logging.info("Splitting dataset into training, validation, and test sets...")
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
logging.info(f"Training set size: {len(train_data)}")
logging.info(f"Validation set size: {len(val_data)}")
logging.info(f"Test set size: {len(test_data)}")

# Step 3: Initialize vectorizer and transform names
logging.info("Initializing vectorizer and transforming training data...")
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
X_train = vectorizer.fit_transform(train_data['name'])
logging.info("Training data transformed.")

logging.info("Transforming validation and test data...")
X_val = vectorizer.transform(val_data['name'])
X_test = vectorizer.transform(test_data['name'])
logging.info("Validation and test data transformation complete.")

# Encode target variables
y_train = train_data['gender']
y_val = val_data['gender']
y_test = test_data['gender']

# Initialize and train the model
logging.info("Initializing and training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
logging.info("Model training complete.")

# Store metrics for final table display
metrics_summary = []

# Define a function to calculate and log metrics
def calculate_metrics(y_true, y_pred, dataset_type="Validation", step=0):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="female", average="binary")
    recall = recall_score(y_true, y_pred, pos_label="female", average="binary")
    f1 = f1_score(y_true, y_pred, pos_label="female", average="binary")
    
    # Log to TensorBoard with a specified step
    writer.add_scalar(f"{dataset_type}/Accuracy", accuracy, step)
    writer.add_scalar(f"{dataset_type}/Precision", precision, step)
    writer.add_scalar(f"{dataset_type}/Recall", recall, step)
    writer.add_scalar(f"{dataset_type}/F1-score", f1, step)

    # Append metrics to summary list for table display
    metrics_summary.append([dataset_type, accuracy, precision, recall, f1])

    # Print and save metrics to file
    with open(metrics_log_path, "a") as log_file:
        log_file.write(f"\n{dataset_type} Metrics:\n")
        log_file.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\n")
    
    logging.info(f"{dataset_type} Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Step 5: Calculate and log metrics for validation and test sets
logging.info("Calculating metrics for validation set...")
y_val_pred = model.predict(X_val)
calculate_metrics(y_val, y_val_pred, dataset_type="Validation", step=1)

logging.info("Calculating metrics for test set...")
y_test_pred = model.predict(X_test)
calculate_metrics(y_test, y_test_pred, dataset_type="Test", step=2)

# Step 6: Save model and vectorizer
logging.info("Saving model and vectorizer to disk...")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
logging.info(f"Model saved to {model_path}")
logging.info(f"Vectorizer saved to {vectorizer_path}")
logging.info(f"Metrics logged to {metrics_log_path}")

# Print summary table
print("\nFinal Metrics Summary:")
print(tabulate(metrics_summary, headers=["Dataset", "Accuracy", "Precision", "Recall", "F1-Score"], floatfmt=".4f"))

# Close TensorBoard writer
writer.close()
logging.info("Training and logging completed.")