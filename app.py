import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score,precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)

import warnings 
warnings.filterwarnings('ignore')

model = joblib.load('pkl/logistic_regression_model.pkl')

# Set page configuration
st.set_page_config(page_title="Bank Marketing Classifier", layout="centered")

# App title and description
st.title("📊 Bank Marketing Classification App")
st.write("Upload **TEST DATA ONLY** (CSV format)")

# File uploader for test CSV
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

#
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

threshold_choice = st.selectbox(
    "Select Threshold",
    [
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7
    ]
)

def calc_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    return {
        "Confusion Matrix": cm,
        "Accuracy": accuracy,
        "AUC-ROC": auc_roc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file, sep=',')

        # Preprocess the test data
        # Assuming the same preprocessing steps as training
        X_test = test_data.drop(columns=['y'])
        y_test = test_data['y'].map({'yes': 1, 'no': 0})

        # Make predictions based on selected model call this method predict_logistic_regression from logistic_model.py
        
        if model_choice == "Logistic Regression":
            y_predproba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_predproba >= threshold_choice).astype(int)
            metrics = calc_metrics(y_test,y_pred)
        
        #display evaluation metrics
            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {metrics['accuracy']:.2f}")
            st.write(f"Precision: {metrics['precision']:.2f}")
            st.write(f"Recall: {metrics['recall']:.2f}")
            st.write(f"F1 Score: {metrics['f1_score']:.2f}")
        #display confusion metrics
            st.subheader("Confusion Matrix")
            st.write(metrics['confusion_matrix'])

    except Exception as e:
        st.error(f"Error processing file: {e}")