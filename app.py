import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

st.title("Credit Card Default Prediction – ML Models Comparison")
st.write(
    """
    This Streamlit application demonstrates multiple machine learning classification models
    applied to the Credit Card Default dataset.
    Users can upload test data, select a model, and view evaluation metrics and confusion matrix.
    """
)

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the test dataset CSV file to proceed.")
    st.stop()

test_df = pd.read_csv(uploaded_file)
st.write("Preview of uploaded data:")
st.dataframe(test_df.head())

X = test_df.drop("default.payment.next.month", axis=1)
y = test_df["default.payment.next.month"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

st.header("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a model",
    list(models.keys())
)

model = models[model_name]

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": roc_auc_score(y, y_prob),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1 Score": f1_score(y, y_pred),
    "MCC": matthews_corrcoef(y, y_pred)
}

st.subheader("Evaluation Metrics")

metrics_df = pd.DataFrame(metrics, index=["Score"]).T
st.table(metrics_df)

st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.subheader("Download Test Dataset")

csv = test_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download test data CSV",
    data=csv,
    file_name="test_data.csv",
    mime="text/csv"
)

