# Customer Intelligence & Predictive Retention Platform

## AI Data Analytics Internship — Project 2

**Developed by:** Muthulakshmi T  
**Project Type:** Machine Learning, Customer Analytics & Predictive Retention  
**Programming Language:** Python

---

## 1. Live Project

**Live Dashboard:**  
https://customer-intelligence-predictive-retention-3p8uxfaxovbsbteplwu.streamlit.app/

**GitHub Repository:**  
https://github.com/miniprakash2002-sudo/Customer-Intelligence-Predictive-Retention

---

## 2. Project Overview

The Customer Intelligence & Predictive Retention Platform is an interactive data analytics application designed to analyze customer behavior, predict customer churn risk, segment customers, compare machine learning models, and generate actionable business recommendations.

The project uses historical telecommunications customer data to identify customers who may discontinue their services. The dashboard helps businesses understand customer risk patterns and prioritize customer retention activities.

The application is developed using Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, and Joblib.

**Important:** This project is an educational demonstration using historical data. Predictions are not guarantees of future customer behavior.

---

## 3. Project Objectives

- Analyze historical customer data.
- Identify customers with a higher predicted churn risk.
- Build and compare machine learning classification models.
- Segment customers using K-Means clustering.
- Visualize customer behavior and churn patterns.
- Identify customers who may need retention attention.
- Provide business recommendations based on analytical results.
- Develop an interactive dashboard for customer intelligence.

---

## 4. Dataset Description

**Dataset Name:** IBM Telco Customer Churn Dataset

**Dataset File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains historical telecommunications customer information.

- Total customer records: 7,043
- Total columns: 21
- Target variable: `Churn`

### Important Features

- Customer ID
- Gender
- Senior citizen status
- Partner and dependents
- Tenure
- Phone service
- Internet service
- Contract type
- Monthly charges
- Total charges
- Payment method
- Churn status

### Data Preparation

The data preparation process includes:

1. Loading the customer dataset.
2. Cleaning and preparing data types.
3. Handling missing or invalid values where required.
4. Preparing input features for machine learning.
5. Separating training and testing data.
6. Preparing customer attributes for segmentation.

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and clustering |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |
| Joblib | Saving and loading trained models |
| Pytest | Automated testing |
| GitHub | Version control and project hosting |
| Streamlit Community Cloud | Dashboard deployment |

---

## 6. Machine Learning Models

The project compares the following classification models:

### 6.1 Dummy Classifier

Used as a baseline to compare the performance of other machine learning models.

### 6.2 Logistic Regression

A classification algorithm used to estimate the probability of customer churn.

### 6.3 Decision Tree Classifier

A tree-based algorithm that learns decision rules from customer data.

### 6.4 Random Forest Classifier

An ensemble algorithm that combines multiple decision trees to make predictions.

### 6.5 Gradient Boosting Classifier

An ensemble learning algorithm that builds models sequentially to improve predictive performance.

### Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision / PR-AUC, where available

 who actually churned. Recall measures the proportion of actual churners identified by the model. F1-score balances precision and recall.

ROC-AUC and Average Precision provide additional information about model ranking performance.

---

## 7. Model Performance Comparison

The following results were recorded during the project evaluation.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Dummy Classifier | 73.46% | 0.00% | 0.00% | 0.00% |
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |
| Decision Tree | 74.10% | 50.77% | 79.14% | 61.86% |
| Random Forest | 75.87% | 53.28% | 73.80% | 61.88% |
| Gradient Boosting | 80.62% | 67.35% | 52.41% | 58.95% |

### Additional Evaluation Results

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Random Forest | 83.67% | 63.93% |
| Gradient Boosting | 84.34% | 66.41% |

Gradient Boosting achieved the highest recorded accuracy at 80.62%. Model selection should also consider recall, precision, and the business cost of missing potential churners.

---

## 8. Customer Segmentation

The project uses K-Means clustering to group customers using the following features:

- Tenure
- Monthly Charges
- Total Charges

The configured workflow uses two customer segments.

Segmentation helps explore customer behavior, compare billing patterns, and support customer targeting. The business meaning of each segment should be validated using the actual segment profiles.

---

## 9. Dashboard Features

### 9.1 Customer Overview
Displays summary information about the customers included in the analysis.

### 9.2 Customer Risk Distribution
Visualizes customers across high-, medium-, and low-risk categories.

### 9.3 Predicted Churn Risk
Displays model-generated risk estimates.

### 9.4 Contract Risk Analysis
Helps explore historical churn patterns across contract types.

### 9.5 Customer Search
Allows users to search using a Customer ID.

### 9.6 Retention Priority List
Highlights customers who may need further review based on predicted risk.

### 9.7 Customer Segmentation
Allows users to explore the configured customer clusters.

### 9.8 Data Export
Supports downloading segmented customer data as a CSV file.

### 9.9 Model Comparison
Displays machine learning model performance for comparison.

### 9.10 Business Recommendations
Provides example retention recommendations based on analytical patterns.

---

## 10. Recorded Dashboard Results

| Metric | Recorded Result |
|---|---:|
| Total Customers | 7,043 |
| High-Risk Customers | 1,514 |
| Average Predicted Risk | 38.3% |
| Historical Churn Rate | 26.5% |

These figures describe the historical demonstration and configured model outputs. Predicted risk is not the same as confirmed future churn.

---

## 11. Project Structure

```text
Customer_Intelligence_Project/
│
├── app.py
├── README.md
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── best_churn_model.joblib
│   ├── customer_segmentation.joblib
│   ├── model_metadata.json
│   ├── train_model.py
│   └── segment_customers.py
│
├── reports/
│   ├── model_comparison.csv
│   ├── segmented_customers.csv
│   ├── segment_profiles.csv
│   └── segmentation_model_comparison.csv
│
└── tests/
    ├── test_project_files.py
    └── test_model_prediction.py
```

Ensure the listed files and folders match your actual repository.

---

## 12. Installation and Execution

### Step 1: Install Python

Check your Python installation in PowerShell.

```powershell
python --version
python -m pip --version
```

### Step 2: Download the Project

```powershell
git clone https://github.com/miniprakash2002-sudo/Customer-Intelligence-Predictive-Retention.git
cd Customer-Intelligence-Predictive-Retention
```

Alternatively, download the repository ZIP from GitHub and extract it.

### Step 3: Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 6: Run the Dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### Step 7: Run Automated Tests

```powershell
python -m pytest -v -p no:cacheprovider
```

The project previously recorded nine passing tests. Run the tests again in your environment to verify the current result.

---

## 13. Testing and Validation

Automated tests are included to check selected project files and model prediction operations.

Recommended validation includes:

- Checking missing and invalid values.
- Verifying model input columns.
- Testing predictions on unseen records.
- Checking customer search and dashboard filters.
- Verifying CSV downloads.
- Monitoring model performance over time.

The previously recorded test result does not establish that every dashboard interaction or production security requirement has been tested.

---

## 14. Business Benefits

The platform demonstrates how customer analytics and machine learning can support business decision-making.

Potential benefits include:

1. Identifying customers who may be at risk of leaving.
2. Prioritizing retention follow-up.
3. Understanding historical customer behavior.
4. Comparing prediction models.
5. Exploring customer segments.
6. Supporting targeted retention campaigns.
7. Presenting analytical findings through an interactive dashboard.

Actual business benefits should be measured through retention campaign outcomes.

---

## 15. Project Limitations

- The application uses historical telecommunications data.
- Historical patterns may not represent future customer behavior.
- Predicted churn risk is not a guarantee that a customer will leave.
- Recommendations require business validation.
- Model performance may vary with different data.
- Production use requires additional monitoring, security, privacy controls, and validation.
- The project does not establish actual financial savings from retention campaigns.

---

## 16. Future Enhancements

- Add more recent customer data.
- Improve model calibration.
- Optimize precision and recall based on retention costs.
- Add explainable AI features.
- Introduce automated model monitoring.
- Connect to a secure customer database.
- Add role-based access control.
- Evaluate retention campaign outcomes.
- Add automated reporting and scheduled updates.

---

## 17. Conclusion

The Customer Intelligence & Predictive Retention Platform demonstrates an end-to-end customer analytics workflow using Python, machine learning, customer segmentation, and interactive visualization.

The project includes data preparation, model comparison, churn risk analysis, K-Means customer segmentation, retention priority exploration, and dashboard-based reporting.

It demonstrates practical applications of data analytics and machine learning to a customer retention problem and provides a foundation for further development.

---

## 18. References

1. IBM Telco Customer Churn Dataset.
2. Pandas Documentation — https://pandas.pydata.org/docs/
3. NumPy Documentation — https://numpy.org/doc/
4. Scikit-learn Documentation — https://scikit-learn.org/stable/
5. Streamlit Documentation — https://docs.streamlit.io/
6. Plotly Documentation — https://plotly.com/python/
7. Pytest Documentation — https://docs.pytest.org/

---

**Project:** Customer Intelligence & Predictive Retention Platform  
**Internship:** AI Data Analytics Internship — Project 2  
**Developed by:** Muthulakshmi T
Accuracy measures overall classification correctness. Precision measures the proportion of predicted churners who actually churned. Recall measures the proportion of actual churners identified
