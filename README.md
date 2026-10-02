# Customer Segmentation & Churn Prediction System

## Project Overview

This project analyzes customer behavior and predicts customer churn using the IBM Telco Customer Churn dataset.

The project combines Exploratory Data Analysis (EDA), customer segmentation using K-Means clustering, and churn prediction using Machine Learning models.

## Objectives

- Analyze customer churn patterns
- Identify important customer characteristics
- Segment customers using K-Means clustering
- Build churn prediction models
- Identify high-risk customers
- Generate insights that can support customer retention analysis

## Dataset

Dataset: IBM Telco Customer Churn Dataset

Total Customers: 7,043

The dataset contains customer demographic information, services, contract details, payment methods, tenure, monthly charges, total charges, and churn status.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Data Loading
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Customer Segmentation
5. Machine Learning Model Development
6. Model Evaluation
7. Churn Risk Prediction
8. High-Risk Customer Analysis
9. Visualization
10. Final Insights

## Customer Segmentation

K-Means clustering was applied using:

- Tenure
- Monthly Charges
- Total Charges

Four customer segments were identified.

## Machine Learning Models

Two classification models were developed:

### Logistic Regression

- Accuracy: 80.62%
- Precision: 65.73%
- Recall: 56.42%
- F1 Score: 60.72%
- ROC-AUC: 84.18%

### Decision Tree

- Accuracy: 73.46%
- Precision: 50.00%
- Recall: 80.75%
- F1 Score: 61.76%
- ROC-AUC: 83.08%

## High-Risk Customer Analysis

The Decision Tree model was used to estimate churn probability and classify customers into:

- Low Risk
- Medium Risk
- High Risk

Risk distribution in the test dataset:

Risk Distribution:

Low Risk: 557 customers
Medium Risk: 433 customers
High Risk: 419 customers

## Key Insights

- 419 customers were identified as high-risk.
- Average tenure of high-risk customers was 13.1 months.
- Average monthly charge of high-risk customers was 76.22.
- Segment 2 showed the highest churn level among the identified customer segments.
- Month-to-month contract customers were prominent among high-risk customers.
- Fiber optic customers were prominent among high-risk customers.
- Electronic check users were prominent among high-risk customers.

## Conclusion

This project demonstrates an end-to-end customer churn analytics workflow using Python and Machine Learning. The analysis combines data cleaning, exploratory data analysis, customer segmentation, churn prediction, and risk profiling.

K-Means clustering was used to identify distinct customer segments, while Logistic Regression and Decision Tree models were developed to predict customer churn. The analysis also identified 419 high-risk customers in the test dataset.

The project provides a structured approach to understanding customer churn patterns and identifying customers who may require further retention analysis.
