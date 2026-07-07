"""
Trains a lightweight baseline classifier on reference.csv to predict
whether a wine is "good" (quality >= 6).
Saves the model, its metrics, and reference feature stats (used later
by the drift check) to model/artifacts/.
"""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
TARGET_COLUMN = "quality_label"


def main():
    df = pd.read_csv("data/reference.csv")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump(model, "model/artifacts/model.joblib")
    with open("model/artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Reference feature stats — used later by the drift check (Phase F)
    reference_stats = {
        col: {"mean": float(X[col].mean()), "std": float(X[col].std())}
        for col in FEATURE_COLUMNS
    }
    with open("model/artifacts/reference_stats.json", "w") as f:
        json.dump(reference_stats, f, indent=2)

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nSaved model.joblib, metrics.json, reference_stats.json to model/artifacts/")


if __name__ == "__main__":
    main()