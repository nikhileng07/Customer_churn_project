import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Title and description
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to check the likelihood of churn.")

# Define input features
def user_input():
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

    # Pack into dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    return input_data

# Get user input
user_df = user_input()

# Preprocessing like you did in notebook
def preprocess_input(df):
    # Same steps from your notebook
    df_encoded = df.copy()
    # Label Encoding
    cols_to_encode = df_encoded.select_dtypes(include='object').columns
    le = joblib.load("label_encoder.pkl")  # Save and reuse encoder from training

    for col in cols_to_encode:
        df_encoded[col] = le[col].transform(df_encoded[col])

    # Scale numeric features
    scaler = joblib.load("scaler.pkl")  # Save and reuse scaler from training
    df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    return df_encoded

# Preprocess and predict
if st.button("Predict Churn"):
    try:
        processed = preprocess_input(user_df)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        st.subheader("🔍 Prediction Result:")
        if prediction == 1:
            st.error(f"❌ This customer is likely to churn. Probability: {probability:.2f}")
        else:
            st.success(f"✅ This customer is not likely to churn. Probability: {probability:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
