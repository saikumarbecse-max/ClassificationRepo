import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score,precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)
from sklearn.metrics import classification_report

import warnings 
warnings.filterwarnings('ignore')

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

def display_metrics(metrics,y_test,y_pred):
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {metrics['Accuracy']:.2f}")
    st.write(f"AUC-ROC: {metrics['AUC-ROC']:.2f}")
    st.write(f"Precision: {metrics['Precision']:.2f}")
    st.write(f"Recall: {metrics['Recall']:.2f}")
    st.write(f"F1 Score: {metrics['F1 Score']:.2f}")
    st.write(f"MCC: {metrics['MCC']:.2f}")

    st.subheader("Confusion Matrix")
    st.write(metrics['Confusion Matrix'])

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file, sep=',')

        # Preprocess the test data
        # Assuming the same preprocessing steps as training
        X_test = test_data.drop(columns=['y'])
        y_test = test_data['y']

        # Make predictions based on selected model call this method predict_logistic_regression from logistic_model.py
        
        if model_choice == "Logistic Regression":
            model = joblib.load('pkl/logistic_model.pkl')
            y_predproba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_predproba >= threshold_choice).astype(int)
            metrics = calc_metrics(y_test,y_pred)

        if metrics:
            display_metrics(metrics,y_test,y_pred)


    except Exception as e:
        st.error(f"Error processing file: {e}")