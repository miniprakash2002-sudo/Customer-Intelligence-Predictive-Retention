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

Accuracy measures overall classification correctness. Precision measures the proportion of predicted churners who actually churned. Recall measures the proportion of actual churners identified
