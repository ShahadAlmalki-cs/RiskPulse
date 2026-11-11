"""Train and persist the RiskPulse Set C credit-risk pipeline.

Products (all under ``models/``):
    risk_pipeline.pkl     Pipeline(imputer + encoder + RandomForest)
    feature_schema.json   feature names, order, categories, missing policy
    metrics.json          hold-out evaluation report

Run from the project root:  python train_model.py
"""

import json
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.data_prep import clean_data, load_raw_data, split_features_target


def build_pipeline() -> Pipeline:
    """Pre-pricing pipeline: impute and encode, then classify."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    categories=[
                        list(config.HOME_OWNERSHIP_OPTIONS),
                        list(config.LOAN_INTENT_OPTIONS),
                        list(config.DEFAULT_ON_FILE_OPTIONS),
                    ],
                    handle_unknown="error",
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, list(config.MODEL_NUMERIC_FEATURES)),
            ("cat", categorical, list(config.MODEL_CATEGORICAL_FEATURES)),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {
        "feature_set": "C (pre-pricing, pre-approval)",
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "default_recall": round(float(tp / (tp + fn)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_train": None,  # filled by caller
        "n_test": int(len(X_test)),
        "test_default_rate": round(float(y_test.mean()), 4),
        "per_class": {
            str(label): {
                "precision": round(float(report[str(label)]["precision"]), 4),
                "recall": round(float(report[str(label)]["recall"]), 4),
                "f1_score": round(float(report[str(label)]["f1-score"]), 4),
                "support": int(report[str(label)]["support"]),
            }
            for label in sorted(y_test.unique())
        },
    }


def main() -> None:
    raw = clean_data(load_raw_data())
    X, y = split_features_target(raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    metrics["n_train"] = int(len(X_train))
    metrics["train_default_rate"] = round(float(y_train.mean()), 4)
    metrics["feature_names"] = list(X.columns)
    metrics["encoded_feature_names"] = [
        name.replace("num__", "").replace("cat__", "")
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]
    metrics["excluded_features"] = {
        "loan_grade": "post-decision pricing label; not available at application",
        "loan_int_rate": "set by the lender at pricing; not available pre-approval",
    }
    metrics["missing_value_policy"] = {
        "numeric": "median imputation (fitted on train only)",
        "categorical": "most_frequent imputation (fitted on train only)",
    }
    metrics["sklearn_version"] = sklearn.__version__
    metrics["python_version"] = platform.python_version()
    metrics["random_state"] = config.RANDOM_STATE

    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1              : {metrics['f1']:.4f}")
    print(f"ROC AUC         : {metrics['roc_auc']:.4f}")
    print(f"PR AUC          : {metrics['pr_auc']:.4f}")
    print(f"Brier           : {metrics['brier']:.4f}")
    print(f"Default recall  : {metrics['default_recall']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, config.PIPELINE_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    schema = {
        "feature_set": "C (pre-pricing, pre-approval)",
        "raw_input_columns": list(X.columns),
        "numeric": list(config.MODEL_NUMERIC_FEATURES),
        "categorical": {
            "person_home_ownership": list(config.HOME_OWNERSHIP_OPTIONS),
            "loan_intent": list(config.LOAN_INTENT_OPTIONS),
            "cb_person_default_on_file": list(config.DEFAULT_ON_FILE_OPTIONS),
        },
        "encoded_feature_names": metrics["encoded_feature_names"],
        "n_encoded_features": len(metrics["encoded_feature_names"]),
        "target": config.TARGET_COLUMN,
        "excluded": metrics["excluded_features"],
        "missing_value_policy": metrics["missing_value_policy"],
        "sklearn_version": sklearn.__version__,
    }
    config.SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"\nPipeline saved to : {config.PIPELINE_PATH}")
    print(f"Schema saved to   : {config.SCHEMA_PATH}")
    print(f"Metrics saved to  : {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
