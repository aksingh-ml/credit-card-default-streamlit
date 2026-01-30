## Live Demo

Streamlit App: https://credit-card-default-app-ml.streamlit.app/

---

# Credit Card Default Prediction using Machine Learning

## 1. Problem Statement

The objective of this project is to build and deploy multiple machine learning
classification models to predict whether a credit card client will default on
the next payment. The project demonstrates an end-to-end machine learning
workflow including data preprocessing, model training, evaluation, and
deployment using a Streamlit web application.

---

## 2. Dataset Description

The dataset used is the **Default of Credit Card Clients** dataset obtained from
the UCI Machine Learning Repository / Kaggle.

- Number of instances: 30,000  
- Number of input features: 24  
- Target variable:  
  - `default.payment.next.month`  
    - 1 = Default  
    - 0 = No Default  

### Feature Groups

**1. Demographic Features**
- LIMIT_BAL: Credit limit
- SEX: Gender of the client
- EDUCATION: Education level
- MARRIAGE: Marital status
- AGE: Age of the client

**2. Repayment Status Features**
- PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6  
These represent the repayment status of the client over the past six months.

**3. Bill Amount Features**
- BILL_AMT1 to BILL_AMT6  
These represent the bill statement amounts for the last six months.

**4. Payment Amount Features**
- PAY_AMT1 to PAY_AMT6  
These represent the amount paid by the client in the past six months.

---

## 3. Models Used and Evaluation Metrics

The following machine learning classification models were implemented and
evaluated on the same dataset:

- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbors (kNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Evaluation Metrics Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| kNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |

---

## 4. Model-wise Observations

**Logistic Regression**  
Logistic Regression achieved good accuracy but relatively low recall,
indicating that while the model is effective in predicting non-default cases,
it misses a significant number of actual defaulters. This behavior is expected
due to class imbalance in the dataset.

**Decision Tree**  
Decision Tree showed lower overall accuracy and AUC compared to ensemble models,
suggesting overfitting to the training data. Although recall improved slightly,
the model did not generalize well on unseen test data.

**k-Nearest Neighbors (kNN)**  
kNN performed moderately well after feature scaling, with balanced precision and
recall compared to Decision Tree. However, its performance was limited due to
sensitivity to distance measures in high-dimensional data.

**Naive Bayes**  
Naive Bayes achieved relatively higher recall among simple models, making it more
effective at identifying defaulters. However, its lower precision indicates a
higher false positive rate, which affects overall accuracy.

**Random Forest (Ensemble)**  
Random Forest achieved one of the highest accuracy and MCC scores, demonstrating
strong generalization capability. By combining multiple decision trees, it
reduced overfitting and handled feature interactions effectively.

**XGBoost (Ensemble)**  
XGBoost achieved the highest AUC score, indicating superior ranking capability
between defaulters and non-defaulters. Its gradient boosting approach allowed it
to capture complex patterns in the data, making it the best-performing model
overall.

---

## 5. Streamlit Application

The Streamlit application allows users to upload test data in CSV format,
select one of the implemented machine learning models, and view evaluation
metrics along with a confusion matrix. The application provides an interactive
interface to demonstrate model performance.

---

## 6. Deployment

The application was deployed on **Streamlit Community Cloud**.

Live Application Link:  
https://credit-card-default-app-ml.streamlit.app/
