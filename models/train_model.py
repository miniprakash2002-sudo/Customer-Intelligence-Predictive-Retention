
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "Churn"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # Validate required target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Encode target
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin(["Yes", "No"])].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1})

    # Remove ID column: it is not a predictive feature
    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
            "PR_AUC_AP": average_precision_score(
                y_test, probabilities
            ),
        }

        results.append(result)
        fitted_models[name] = pipeline

        print(f"\n{name}")
        for metric, value in result.items():
            if metric != "Model":
                print(f"  {metric}: {value:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        by="F1", ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )

    # Select the best model by F1 on the held-out test set
    # for this initial demonstration. Final model selection
    # should use validation or cross-validation instead.
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_models[best_name]

    joblib.dump(
        best_pipeline,
        MODEL_DIR / "best_churn_model.joblib"
    )

    metadata = {
        "model_name": best_name,
        "target": TARGET,
        "decision_threshold": 0.5,
        "positive_class": "Yes",
        "features": X.columns.tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_churn_model.joblib'}")
    print(f"Saved metrics: {REPORT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
