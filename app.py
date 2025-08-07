import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_data
def load_model():
    # No need to load CSV for model — it was just used during training
    model = joblib.load("model.pkl")
    le = joblib.load("encoder.pkl")
    return model, le

model, le = load_model()

st.title("🏦 Loan Approval Prediction App")

st.write("Enter the applicant details below to check loan approval status.")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [360, 120, 180, 240, 300, 60, 84])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

if st.button("Predict"):
    # Collect input
    input_dict = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = le.transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")
        # Simple explanation (can be improved with SHAP or LIME)
        reasons = []
        if credit_history == 0.0:
            reasons.append("- Poor Credit History")
        if applicant_income < 2500:
            reasons.append("- Low Applicant Income")
        if loan_amount > 200:
            reasons.append("- High Loan Amount Requested")
        if self_employed == 'Yes' and applicant_income < 4000:
            reasons.append("- Self-employed with lower income")

        if reasons:
            st.subheader("Possible Reasons:")
            for r in reasons:
                st.write(r)
        else:
            st.write("The application does not meet model approval conditions.")

