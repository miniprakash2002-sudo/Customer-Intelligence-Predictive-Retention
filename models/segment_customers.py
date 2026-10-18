
from pathlib import Path

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    missing_columns = [col for col in FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert numeric fields and handle invalid values
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[FEATURES].copy()

    # Remove rows where all clustering features are missing
    valid_rows = X.notna().any(axis=1)
    df = df.loc[valid_rows].copy()
    X = X.loc[valid_rows].copy()

    # Impute missing values and scale features before clustering
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_scaled = preprocessing.fit_transform(X)

    # Compare several cluster counts using silhouette score
    scores = []
    candidate_models = {}

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(2000, len(X_scaled)),
            random_state=42,
        )

        scores.append({
            "Clusters": k,
            "Silhouette_Score": score,
        })
        candidate_models[k] = model

    score_df = pd.DataFrame(scores)
    score_df.to_csv(
        REPORT_DIR / "segmentation_model_comparison.csv",
        index=False,
    )

    best_k = int(
        score_df.loc[
            score_df["Silhouette_Score"].idxmax(),
            "Clusters",
        ]
    )
    best_model = candidate_models[best_k]

    df["Customer_Segment"] = best_model.labels_ + 1

    # Profile clusters using original business units
    profiles = (
        df.groupby("Customer_Segment")[FEATURES]
        .mean()
        .round(2)
    )
    profiles["Customer_Count"] = (
        df["Customer_Segment"].value_counts()
        .reindex(profiles.index)
    )
    profiles = profiles.reset_index()

    # Create readable, data-driven segment descriptions
    tenure_median = df["tenure"].median()
    charge_median = df["MonthlyCharges"].median()

    def describe_segment(row):
        descriptions = []

        if row["tenure"] <= tenure_median:
            descriptions.append("Shorter-tenure customers")
        else:
            descriptions.append("Longer-tenure customers")

        if row["MonthlyCharges"] >= charge_median:
            descriptions.append("Higher monthly charges")
        else:
            descriptions.append("Lower monthly charges")

        return " | ".join(descriptions)

    profiles["Segment_Profile"] = profiles.apply(
        describe_segment, axis=1
    )

    df.to_csv(
        REPORT_DIR / "segmented_customers.csv",
        index=False,
    )
    profiles.to_csv(
        REPORT_DIR / "segment_profiles.csv",
        index=False,
    )

    joblib.dump(
        {
            "preprocessor": preprocessing,
            "kmeans": best_model,
            "features": FEATURES,
            "cluster_count": best_k,
        },
        MODEL_DIR / "customer_segmentation.joblib",
    )

    print("\nSilhouette score comparison:")
    print(score_df.round(4).to_string(index=False))
    print(f"\nSelected number of clusters: {best_k}")
    print("\nSegment profiles:")
    print(profiles.to_string(index=False))
    print("\nSaved files:")
    print("models/customer_segmentation.joblib")
    print("reports/segmented_customers.csv")
    print("reports/segment_profiles.csv")
    print("reports/segmentation_model_comparison.csv")


if __name__ == "__main__":
    main()
