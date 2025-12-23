# Autism Prediction in Adults using Machine Learning

A machine learning–based system for early prediction of Autism Spectrum Disorder (ASD) in adults, using screening questionnaire data and demographic features.
# Problem Statement

Autism Spectrum Disorder in adults often remains undiagnosed due to subtle symptoms, lack of awareness, and limited access to trained professionals. Traditional diagnosis methods are time-consuming, subjective, and resource-intensive.

This project aims to build an automated machine learning model that can assist in early ASD screening for adults, supporting clinicians and individuals with faster, data-driven insights.
# Objectives

Develop a binary classification system (ASD / Non-ASD)

Compare multiple machine learning algorithms

Handle class imbalance and data quality issues

Identify the most reliable model for ASD prediction

Provide interpretability using feature importance

Enable practical deployment through a web application

# Dataset

Name: Autism Screening on Adults

Source: UCI Machine Learning Repository

Records: 704

Features: 21 (screening scores + demographic attributes)

Target Variable: Class/ASD (Yes / No)

Key Features: 

A1Score to A10Score (screening questionnaire responses)

Age, Gender, Ethnicity

Jaundice at birth

Family history of autism

Country of residence

Previous screening information

# Methodology

Data cleaning and missing value handling

Label encoding of categorical variables

Outlier detection and correction

Exploratory Data Analysis (EDA)

Train-test split (80% training, 20% testing)

Class imbalance handling using SMOTE

Model training and comparison

Hyperparameter tuning using GridSearchCV

Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC

Best model selection and deployment preparation

# Models Implemented

Logistic Regression

Decision Tree Classifier

Support Vector Machine (SVM)

Random Forest Classifier

Gaussian Naive Bayes

# Model Optimization

Cross-validation: Stratified K-Fold (5 folds)

Hyperparameter tuning: GridSearchCV

Feature scaling: StandardScaler (for LR, SVM, GNB)

# Results
| Model               | Accuracy | ROC-AUC    |
| ------------------- | -------- | ---------- |
| Logistic Regression | 93%      | 1.00       |
| Decision Tree       | 82%      | 0.86       |
| SVM                 | 93%      | 0.98       |
| Gaussian NB         | 94%      | 0.99       |
| **Random Forest**   | **95%**  | **0.9972** |

# Best Model

Random Forest Classifier was selected as the final model due to:

Highest accuracy (95%)

Near-perfect ROC-AUC score

Balanced precision and recall

Strong generalization and reduced overfitting
# Feature Importance (Insights)

Most influential predictors:

A9_Score

A5_Score

A6_Score

A3_Score

Age

Country of residence

Least impactful features:

Used_app_before

Jaundice

# Deployment

Final trained model saved using pickle (.pkl)

Integrated into a Flask-based web application

Frontend built using HTML & CSS

Users input screening scores and demographic details to receive predictions

# Tech Stack

Python

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

SMOTE

Flask

HTML, CSS

# Future Improvements

Clinical validation with real patient data

Use explainability tools like SHAP or LIME

Expand dataset for better generalization

Develop a mobile application version

Improve bias handling and fairness

# Authors & Collaborators

This project was developed as part of a **Summer Research Internship (Campus Research Project)**.

- **Guggilam Leela Naga Sai Sri Saketh**  
- **Seshagiri Bharadwaj Sai**  
- **Annam Surya Teja**  
- **Hrushikesh Bhaskar Gopale**

**Department:** Computer Science and Engineering  
**Institution:** SRM University – AP  
**Internship Duration:** June 2025 – July 2025

### Faculty Supervisor
- **Mr. B. L. V. Siva Rama Krishna**  
  Assistant Professor, Department of CSE  
  SRM University – AP




