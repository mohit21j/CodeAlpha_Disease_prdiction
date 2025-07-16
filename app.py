
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

# Select model
model_options = ["Random Forest", "Logistic Regression", "Support Vector Machine", "XGBoost"]
model_choice = st.sidebar.selectbox("Select ML Model", model_options)

# Load selected model
model_path = f"{model_choice}.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"Model file '{model_path}' not found.")
    st.stop()

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

