
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

## 8. Project Structure## 8. Project Structure

```text
Customer_Intelligence_Project/
├── app.py
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
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

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install required packages

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Run the dashboard

```powershell
python -m streamlit run app.py
```

### Step 5: Run automated tests

```powershell
python -m pytest -v -p no:cacheprovider
```








```
