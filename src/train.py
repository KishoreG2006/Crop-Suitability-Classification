import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
# Adjust sys.path to allow imports from project root
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_data, split_data, get_preprocessing_pipeline
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and split data
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(df)

# Define models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "decision_tree": DecisionTreeClassifier(random_state=42),
}

results = {}
best_accuracy = 0.0
best_model_name = None
best_pipeline = None

for name, model in models.items():
    # Build pipeline: scaling + model (scaling needed for LR & KNN)
    pipeline = Pipeline([
        ("scaler", get_preprocessing_pipeline().named_steps["scaler"]),
        ("clf", model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }
    print(f"Model: {name}\n  Accuracy: {acc:.4f}\n  Precision: {precision:.4f}\n  Recall: {recall:.4f}\n  F1: {f1:.4f}\n")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_pipeline = pipeline

# Save the best model
model_path = os.path.join(MODEL_DIR, "crop_model.pkl")
joblib.dump(best_pipeline, model_path)
print(f"Best model '{best_model_name}' saved to {model_path} (accuracy={best_accuracy:.4f})")

# Optional: write a summary CSV of results
summary_df = pd.DataFrame.from_dict(
    {k: {"accuracy": v["accuracy"], "precision": v["precision"], "recall": v["recall"], "f1_score": v["f1_score"]} for k, v in results.items()},
    orient="index",
)
summary_path = os.path.join(MODEL_DIR, "model_comparison.csv")
summary_df.to_csv(summary_path)
print(f"Model comparison saved to {summary_path}")
