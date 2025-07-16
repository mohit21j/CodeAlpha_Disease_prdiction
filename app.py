import streamlit as st
import numpy as np
import pickle
import os

# Load column names (features expected by model)
with open("columns.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Page title and sidebar
st.set_page_config(page_title="Disease Prediction", layout="centered")
st.title(" Disease Prediction App")
st.markdown("Choose a model, enter patient data, and get predictions.")

# Dropdown options mapped to model filenames
model_files = {
    "Logistic_Regression": "Logistic_Regression.pkl",
    "Support_Vector_Machine": "Support_Vector_Machine.pkl",
    "Random_Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

# Model selection dropdown
selected_model_name = st.selectbox("Select ML Model", list(model_files.keys()))
model_filename = model_files[selected_model_name]

# Load selected model
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# Sidebar input form
st.sidebar.header("Enter Patient Data")
user_input = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.2f")
    user_input.append(val)

# Predict button
if st.button("Predict"):
    input_data = np.array([user_input])
    prediction = model.predict(input_data)[0]

    # Use predict_proba only if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][int(prediction)]
    else:
        probability = 1.0  # fallback for models that don't support predict_proba

    if prediction == 1:
        st.error(f" Disease Detected! ({probability*100:.2f}% confidence)")
    else:
        st.success(f" No Disease Detected ({probability*100:.2f}% confidence)")

