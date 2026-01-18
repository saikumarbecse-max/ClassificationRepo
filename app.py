import streamlit as st
import joblib
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')

from logistic_model import predict_logistic_regression,calc_metrics

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

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file, sep=';')

        # Preprocess the test data
        # Assuming the same preprocessing steps as training
        X_test = test_data.drop(columns=['y'])
        y_test = test_data['y'].map({'yes': 1, 'no': 0})

        # Make predictions based on selected model call this method predict_logistic_regression from logistic_model.py
        
        if model_choice == "Logistic Regression":
            y_pred = predict_logistic_regression(model, X_test, threshold=0.5)
            metrics = calc_metrics(y_test, y_pred)
        
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