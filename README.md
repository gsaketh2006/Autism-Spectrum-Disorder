# Autism Prediction in Adults using Machine Learning

A machine learning–based screening system for early prediction of Autism Spectrum Disorder (ASD) in adults, using questionnaire responses and demographic features.

**Live Demo:** https://asd-in-adults-prediction.onrender.com/

---

## Problem Statement
Autism Spectrum Disorder in adults often remains undiagnosed due to:
- Subtle symptoms  
- Lack of awareness  
- Limited access to trained professionals  

Traditional diagnosis methods are **time-consuming, subjective, and resource-intensive**.  
This project proposes an **automated ML-based screening tool** to assist clinicians and individuals with **faster, data-driven insights**.

---

## Objectives
- Build a **binary classification system** (ASD / Non-ASD)
- Compare multiple **machine learning algorithms**
- Handle **class imbalance** and **data quality issues**
- Identify the **most reliable model** for ASD prediction
- Provide **model interpretability** using feature importance
- Enable **real-world deployment** through a web application

---

## Dataset
- **Name:** Autism Screening on Adults  
- **Source:** UCI Machine Learning Repository  
- **Total Records:** 704  
- **Total Features:** 21  
- **Target Variable:** `Class/ASD` (Yes / No)

### Key Features
- **Screening Scores:** A1Score – A10Score  
- **Demographics:** Age, Gender, Ethnicity  
- **Medical History:** Jaundice at birth, Family history of autism  
- **Geographical:** Country of residence  
- **Usage Info:** Previous screening information  

---

## Methodology
- Data cleaning and missing value handling  
- Label encoding of categorical variables  
- Outlier detection and correction  
- Exploratory Data Analysis (EDA)  
- Train–test split (80% training, 20% testing)  
- Class imbalance handling using **SMOTE**  
- Model training and comparison  
- Hyperparameter tuning using **GridSearchCV**  
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC–AUC  
- Best model selection and deployment preparation  

---

## Models Implemented
- Logistic Regression  
- Decision Tree Classifier  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Gaussian Naive Bayes  

---

## Model Optimization
- **Cross-validation:** Stratified K-Fold (5 folds)  
- **Hyperparameter tuning:** GridSearchCV  
- **Feature scaling:** StandardScaler  
  *(applied to LR, SVM, and Gaussian NB)*  

---

## Results
| Model | Accuracy | ROC-AUC |
|------|----------|---------|
| Logistic Regression | 93% | 1.00 |
| Decision Tree | 82% | 0.86 |
| SVM | 93% | 0.98 |
| Gaussian Naive Bayes | 94% | 0.99 |
| **Random Forest** | **95%** | **0.9972** |

---

## Best Model Selection
**Random Forest Classifier** was selected as the final model due to:
- Highest accuracy (**95%**)  
- Near-perfect ROC–AUC score  
- Balanced precision and recall  
- Strong generalization with reduced overfitting  

---

## Feature Importance (Insights)
<img width="1152" height="672" alt="image" src="https://github.com/user-attachments/assets/f8f9754d-c4c0-497c-91ff-7436d62dee87" />

### Most Influential Predictors
- A9_Score  
- A5_Score  
- A6_Score  
- A3_Score  
- Age  
- Country of residence  

### Least Impactful Features
- Used_app_before  
- Jaundice  

---

## Deployment
- Final trained model saved using **pickle (.pkl)**  
- Integrated into a **Flask-based web application**  
- Frontend developed using **HTML & CSS**  
- Users input screening scores and demographic details to receive predictions  

---

## Tech Stack
- **Programming:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **ML Techniques:** SMOTE  
- **Backend:** Flask  
- **Frontend:** HTML, CSS  

---

## Future Improvements
- Clinical validation with real patient data  
- Model explainability using **SHAP / LIME**  
- Expand dataset for better generalization  
- Develop a **mobile application version**  
- Improve bias handling and fairness  

---

## Authors & Collaborators
This project was developed as part of a  
**Summer Internship (Campus Research Project Under Faculty)**.

- **Guggilam Leela Naga Sai Sri Saketh**  
- **Seshagiri Bharadwaj Sai**  
- **Annam Surya Teja**  
- **Hrushikesh Bhaskar Gopale**

**Department:** Computer Science and Engineering  
**Institution:** SRM University – AP  
**Internship Duration:** June 2025 – July 2025  

---

## Faculty Supervisor
- **Mr. B. L. V. Siva Rama Krishna**  
  Assistant Professor, Department of CSE  
  SRM University – AP  

---
