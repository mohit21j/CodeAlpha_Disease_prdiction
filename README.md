# CodeAlpha_Task

**Disease Prediction Web App using Machine Learning**

This project predicts the likelihood of diabetes using multiple ML models trained on a medical dataset. Built with Streamlit.

# Dataset
Dataset used: "diabetes.csv"

# Models Used
Trained on:
- Random Forest
- Logistic Regression
- SVM (Support Vector Machine)
- XGBoost

Each model is trained and saved separately as `.pkl` files for easy use.

##  Files Included

| File Name :- Purpose |

 `model_training.ipynb` :- Jupyter notebook to train and save ML models 
 `RandomForest.pkl`, `LogisticRegression.pkl`, etc. :- Saved trained models 
 `columns.pkl` :- Saved feature names (used by Streamlit UI) 
 `app.py` :- Streamlit app that loads models and predicts 
 `diabetes.csv` :- Dataset file 
 `README.md` :- This file 


## How to Run the App Locally
> You must install `streamlit`, `sklearn`, and `xgboost`.

