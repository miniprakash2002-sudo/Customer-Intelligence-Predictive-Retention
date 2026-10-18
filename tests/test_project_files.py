
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists():
    assert (ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv").exists()


def test_model_comparison_exists():
    assert (ROOT / "reports" / "model_comparison.csv").exists()


def test_segmentation_outputs_exist():
    assert (ROOT / "reports" / "segmented_customers.csv").exists()
    assert (ROOT / "reports" / "segment_profiles.csv").exists()


def test_dataset_has_expected_columns():
    data = pd.read_csv(
        ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    assert len(data) > 0
    assert "Churn" in data.columns
    assert "customerID" in data.columns


def test_model_comparison_metrics_are_valid():
    results = pd.read_csv(ROOT / "reports" / "model_comparison.csv")

    required = {
        "Model", "Accuracy", "Precision", "Recall",
        "F1", "ROC_AUC", "PR_AUC_AP"
    }

    assert required.issubset(results.columns)
    assert not results.empty

    for metric in required - {"Model"}:
        assert results[metric].between(0, 1).all(), (
            f"Invalid metric values in {metric}"
        )
