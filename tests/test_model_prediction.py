
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_trained_model_exists():
    assert MODEL_PATH.exists(), "Trained model file is missing"


def test_model_can_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predicts_customer_churn():
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    # Use one real customer row and remove the target column
    customer = data.drop(columns=["Churn"]).head(1)

    prediction = model.predict(customer)

    assert len(prediction) == 1
    assert str(prediction[0]) in {"Yes", "No", "0", "1"}


def test_model_predict_proba_if_supported():
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    customer = data.drop(columns=["Churn"]).head(1)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer)

        assert probabilities.shape[0] == 1
        assert 0 <= probabilities[0].max() <= 1
