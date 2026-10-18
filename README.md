
# Customer Intelligence & Predictive Retention Platform

## Project 2 — AI Data Analytics Internship

## 1. Project Overview

The Customer Intelligence & Predictive Retention Platform is a machine learning-based analytics application designed to analyze customer behavior, predict potential customer churn, segment customers, and support customer retention strategies.

The project uses Python, Pandas, Scikit-learn, and Streamlit to build an interactive dashboard for customer analytics and predictive retention.

## 2. Project Objectives

- Analyze historical customer data.
- Predict customer churn using classification algorithms.
- Compare machine learning models using evaluation metrics.
- Segment customers using K-Means clustering.
- Identify high-risk customers for retention prioritization.
- Provide business-oriented retention recommendations.
- Present insights through an interactive dashboard.

## 3. Dataset

**Dataset:** Telco Customer Churn

**Source:** IBM Telco Customer Churn sample dataset, commonly distributed through public machine learning repositories.

**Records:** 7,043 customers  
**Columns:** 21

Important features include:
- CustomerID
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- InternetService
- Contract
- MonthlyCharges
- TotalCharges
- Churn

The dataset contains historical customer information and a Churn target variable.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Pytest

## 5. Machine Learning Models

The project compares the following classification models:

1. Dummy Baseline
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision

The model comparison results are saved in `reports/model_comparison.csv`.

## 6. Customer Segmentation

K-Means clustering is used to group customers according to selected numerical features:

- Tenure
- Monthly Charges
- Total Charges

The segmentation pipeline uses missing-value imputation and feature scaling.

Generated files:
- `models/customer_segmentation.joblib`
- `reports/segmented_customers.csv`
- `reports/segment_profiles.csv`
- `reports/segmentation_model_comparison.csv`

The current segmentation run produced two customer segments.

## 7. Dashboard Features

The Streamlit dashboard includes:

- Customer analytics summary
- Predicted customer risk distribution
- Predicted risk by contract type
- Customer retention priority list
- Risk category filters
- Customer ID search
- Customer segmentation analysis
- Segment exploration and CSV download
- Machine learning model comparison
- Individual model evaluation metrics

## 8. Project Structure

```text
Customer_Intelligence_Project/
├── app.py
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── best_churn_model.joblib
│   ├── customer_segmentation.joblib
│   ├── model_metadata.json
│   ├── train_model.py
│   └── segment_customers.py
├── reports/
│   ├── model_comparison.csv
│   ├── segmented_customers.csv
│   ├── segment_profiles.csv
│   └── segmentation_model_comparison.csv
└── tests/
    ├── test_project_files.py
    └── test_model_prediction.py
```

## 9. Installation and Execution

### Step 1: Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install required packages

```powershell
python -m pip install pandas numpy scikit-learn streamlit plotly joblib pytest
```

### Step 3: Run the dashboard

```powershell
python -m streamlit run app.py
```

### Step 4: Run automated tests

```powershell
python -m pytest -v -p no:cacheprovider
```

## 10. Testing and Validation

The current automated test suite contains nine tests.

The tests verify:
- Dataset availability
- Required dataset columns
- Model comparison report availability
- Segmentation output availability
- Evaluation metric validity
- Trained model loading
- Customer churn prediction
- Prediction probability output, when supported

**Recorded test result:** 9 tests passed.

These tests verify selected project files and model operations. They do not represent complete production or user-interface testing.

## 11. Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 75.87% | 53.28% | 73.80% | 61.88% |
| Decision Tree | 74.10% | 50.77% | 79.14% | 61.86% |
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |
| Gradient Boosting | 80.62% | 67.35% | 52.41% | 58.95% |
| Dummy Baseline | 73.46% | 0.00% | 0.00% | 0.00% |

Gradient Boosting achieved the highest accuracy in this recorded comparison. Random Forest achieved the highest F1-score, while Decision Tree achieved the highest recall.

Model selection should consider the business objective, especially the cost of missing customers who may churn.

## 12. Business Benefits

- Helps prioritize customers with higher predicted churn risk.
- Supports data-driven retention planning.
- Provides visibility into customer segments.
- Makes model performance easier to compare.
- Presents analytical findings through a user-friendly dashboard.

The retention recommendations in this academic project are rule-based examples, not verified interventions or guaranteed outcomes.

## 13. Limitations

- The dataset contains historical data, not live customer activity.
- Predictions depend on the available features and trained model.
- Predicted churn risk does not guarantee that a customer will leave.
- Model performance may change with new customer populations.
- Segments require business interpretation before operational use.
- Production deployment would require further monitoring, validation, security, and privacy controls.

## 14. Conclusion

The Customer Intelligence & Predictive Retention Platform demonstrates an end-to-end analytics workflow combining data preparation, machine learning classification, model evaluation, customer segmentation, automated testing, and interactive visualization.

The project provides a foundation for understanding customer churn and exploring potential retention strategies using historical data.

## 15. Future Enhancements

- API-based churn prediction
- Model threshold tuning
- Cross-validation and improved model selection
- Explainable AI using feature importance or SHAP
- Automated data quality checks
- Model monitoring and retraining
- Secure deployment with access controls

---

**Project:** Customer Intelligence & Predictive Retention Platform  
**Program:** AI Data Analytics Internship  
**Application:** Python and Streamlit
