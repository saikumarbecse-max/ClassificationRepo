import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

threshold_choice = st.sidebar.slider(
    "Select Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
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

def display_metrics(metrics,y_test,y_pred,y_predproba):

    st.subheader("📈 Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['Precision']:.3f}")
    col3.metric("Recall", f"{metrics['Recall']:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
    col5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.3f}")
    col6.metric("MCC", f"{metrics['MCC']:.3f}")

    st.markdown("---")

    left_col, right_col = st.columns(2)
    # ---------- CONFUSION MATRIX (PLOT) ----------
    with left_col:
        st.subheader("🔍 Confusion Matrix")

        cm = metrics["Confusion Matrix"]
        fig, ax = plt.subplots(figsize=(4, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

    # ---------- CLASSIFICATION REPORT ----------
    with right_col:
        st.subheader("📄 Classification Report")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df, use_container_width=True)

        csv = report_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download Classification Report",
            csv,
            "classification_report.csv",
            "text/csv"
        )

    st.markdown("---")

    # =======================
    # PREDICTION SUMMARY
    # =======================

    st.subheader("📊 Prediction Summary")

    pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Probability": y_predproba
    })

    col7, col8 = st.columns([1, 2])

    with col7:
        st.write("Prediction Counts")
        st.bar_chart(pred_df["Predicted"].value_counts())

    with col8:
        st.write("Sample Predictions")
        st.dataframe(pred_df.head(10), use_container_width=True)



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
            metrics = calc_metrics(y_test,y_pred,y_predproba)

        if metrics:
            display_metrics(metrics,y_test,y_pred)


    except Exception as e:
        st.error(f"Error processing file: {e}")