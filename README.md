# 🏦 Home Credit Default Risk Prediction

## 📌 Project Overview

This project aims to predict whether a customer is likely to default on a loan.

The system uses Machine Learning classification models to analyze customer financial and demographic information and predict the probability of loan default.

The project also includes a Streamlit web application that allows users to enter customer information and receive a default risk prediction.

---

## 🎯 Problem Statement

Financial institutions need to identify customers who may have difficulty repaying their loans.

The main challenge is that the dataset is imbalanced, meaning that the number of non-default customers is much larger than the number of default customers.

Therefore, the model should not rely only on Accuracy. Other evaluation metrics such as Recall, F1-Score, Precision, and ROC-AUC are also considered.

---

## 🎯 Project Objectives

- Clean and preprocess the dataset.
- Analyze customer financial information.
- Handle missing values and data quality problems.
- Handle class imbalance.
- Select important features.
- Train multiple Machine Learning classification models.
- Compare model performance.
- Tune the best-performing model.
- Build a prediction interface using Streamlit.
- Predict the probability of loan default for a new customer.

---

## 📊 Dataset

The project is based on the Home Credit Default Risk dataset.

The dataset contains information about customers such as:

- Income
- Credit amount
- Annuity
- Age
- Employment information
- Registration information
- External credit scores
- Credit-to-income ratios
- Previous application information

### Target Variable

```text
0 → Non-Default
1 → Default# NTI-Final-Project
