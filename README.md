\## Live Demo

Streamlit App: https://credit-card-default-app-ml.streamlit.app/





\# Credit Card Default Prediction using Machine Learning



\## 1. Problem Statement

\[You will write this – I will help]



\### Dataset Description



The dataset used is the "Default of Credit Card Clients" dataset obtained from the UCI / Kaggle repository.

It contains 30,000 instances and 24 input features along with one binary target variable.



The target variable is:

\- default.payment.next.month (1 = Default, 0 = No Default)



The input features can be grouped as follows:



1\. Demographic Features:

\- LIMIT\_BAL: Credit limit

\- SEX: Gender of the client

\- EDUCATION: Education level

\- MARRIAGE: Marital status

\- AGE: Age of the client



2\. Repayment Status Features:

\- PAY\_0, PAY\_2, PAY\_3, PAY\_4, PAY\_5, PAY\_6

These features represent the repayment status of the client over the past six months.



3\. Bill Amount Features:

\- BILL\_AMT1 to BILL\_AMT6

These represent the bill statement amounts for the last six months.



4\. Payment Amount Features:

\- PAY\_AMT1 to PAY\_AMT6

These represent the amount paid by the client in the past six months.



\## 3. Models Used and Evaluation Metrics

\### Model-wise Observations



| ML Model | Observation about model performance |

|---------|------------------------------------|

| Logistic Regression | |

| Decision Tree | |

| kNN | |

| Naive Bayes | |

| Random Forest | |

| XGBoost | |



Logistic Regression:

Logistic Regression achieved good accuracy but relatively low recall, indicating that while the model is effective in predicting non-default cases, it misses a significant number of actual defaulters. This is expected due to class imbalance in the dataset.



Decision Tree:

Decision Tree showed lower overall accuracy and AUC compared to ensemble models, suggesting overfitting to the training data. Although recall improved slightly, the model did not generalize well on unseen test data.



k-Nearest Neighbors (kNN):

kNN performed moderately well after feature scaling, with balanced precision and recall compared to Decision Tree. However, its performance was limited due to sensitivity to distance measures in high-dimensional data.



Naive Bayes:

Naive Bayes achieved relatively higher recall among simple models, making it more effective at identifying defaulters. However, its lower precision indicates a higher false positive rate, which affects overall accuracy.



Random Forest (Ensemble):

Random Forest achieved one of the highest accuracy and MCC scores, demonstrating strong generalization capability. By combining multiple decision trees, it reduced overfitting and handled feature interactions effectively.



XGBoost (Ensemble):

XGBoost achieved the highest AUC score, indicating superior ranking capability between defaulters and non-defaulters. Its gradient boosting approach allowed it to capture complex patterns in the data, making it one of the best-performing models overall.





\## 4. Observations

\[Model-wise observations – later]



\## 5. Streamlit Application

\[Short description – later]



\## 6. Deployment

\[Streamlit Cloud details – later]

