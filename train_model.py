"""Train and persist the RiskPulse credit-risk artifacts.

Outputs (all under ``models/``):
    risk_model.pkl  trained RandomForestClassifier
    scaler.pkl      StandardScaler fitted on the training features
    metrics.json    hold-out evaluation report used by the dashboard

Run from the project root:

    python train_model.py
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_prep import clean_data, load_raw_data, split_features_target


def load_training_data() -> pd.DataFrame:
    """Prefer the cleaned CSV; fall back to cleaning the raw dataset in memory."""
    if config.CLEAN_DATASET_PATH.exists():
        return pd.read_csv(config.CLEAN_DATASET_PATH)
    return clean_data(load_raw_data())


def main() -> None:
    X, y = split_features_target(load_training_data())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    # Fit the scaler on the training split only to avoid leaking test statistics.
    scaler = StandardScaler().fit(X_train)

    predictions = model.predict(X_test)
    default_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, default_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification report:")
    report = classification_report(y_test, predictions, output_dict=True)
    print(classification_report(y_test, predictions))

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(roc_auc), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_default_rate": round(float(y_train.mean()), 4),
        "features": [str(column) for column in model.feature_names_in_],
        # The classifier's own class order is used below to read probabilities.
        "classes": [str(label) for label in model.classes_],
        "default_class_index": list(model.classes_).index(1),
        "per_class": {
            str(label): {
                "precision": round(float(report[str(label)]["precision"]), 4),
                "recall": round(float(report[str(label)]["recall"]), 4),
                "f1_score": round(float(report[str(label)]["f1-score"]), 4),
                "support": int(report[str(label)]["support"]),
            }
            for label in model.classes_
        },
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nTrained model saved to: {config.MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    print(f"Metrics saved to: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
