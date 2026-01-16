import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Online Shoppers Purchase Prediction", layout="wide")

st.title("Online Shoppers Purchasing Intention Prediction")
st.write(
    """
    This application predicts whether an online shopper will make a purchase
    based on browsing behavior.  
    Users can upload test data, select a trained ML model, and view predictions
    along with evaluation metrics.
    """
)

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl"),
    }
    return models

@st.cache_data
def load_metrics():
    return pd.read_csv("model/model_results.csv")

models = load_models()
metrics_df = load_metrics()

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

selected_model = models[model_name]


st.subheader("Model Evaluation Metrics")

model_metrics = metrics_df[metrics_df["Model"] == model_name]

st.dataframe(model_metrics, use_container_width=True)


st.subheader("Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(test_df.head())


    # Encode categorical columns
    categorical_cols = ['Month', 'VisitorType', 'Weekend']

    for col in categorical_cols:
        if col in test_df.columns:
            le = LabelEncoder()
            test_df[col] = le.fit_transform(test_df[col])

    # Separate features and target if present
    if 'Revenue' in test_df.columns:
        X_test = test_df.drop('Revenue', axis=1)
        y_true = test_df['Revenue'].astype(int)
    else:
        X_test = test_df
        y_true = None

    # Scale features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)


    st.subheader("Prediction Results")

    # Choose scaled or unscaled input
    if model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
        y_pred = selected_model.predict(X_test)
    else:
        y_pred = selected_model.predict(X_test_scaled)

    st.write("Prediction counts:")
    st.write(pd.Series(y_pred).value_counts())


    if y_true is not None:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))


