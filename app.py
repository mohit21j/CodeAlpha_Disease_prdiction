
import streamlit as st
import numpy as np
import pickle
import os

# Load column names
with open("columns.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Title and sidebar
st.set_page_config(page_title="Disease Prediction", layout="centered")
st.title(" Disease Prediction App")
st.markdown("Choose a model, enter patient data, and get predictions.")

# Dropdown options mapped to filenames
model_files = {
    "Logistic_Regression": "logistic_model.pkl",
    "Support_Vector_Machine": "svm_model.pkl",
    "Random_Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# Streamlit dropdown
selected_model_name = st.selectbox("Select ML Model", list(model_files.keys()))

# Get actual file name from dictionary
model_filename = model_files[selected_model_name]

# Load model
with open(model_filename, 'rb') as f:


# Input fields dynamically from feature names
st.sidebar.header("Enter Patient Data")
user_input = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.2f")
    user_input.append(val)

# Predict
if st.button("🔍 Predict"):
    input_data = np.array([user_input])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][int(prediction)]

    if prediction == 1:
        st.error(f"Disease Detected! ({probability*100:.2f}% confidence)")
    else:
        st.success(f"No Disease Detected ({probability*100:.2f}% confidence)")

