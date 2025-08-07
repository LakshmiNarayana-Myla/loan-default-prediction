import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and prepare model & encoders
@st.cache_resource
def load_model():
    df = pd.read_csv("data/train.csv")

    # Drop ID
    df.drop("Loan_ID", axis=1, inplace=True)

    # Fill missing
    for col in ["Gender", "Married", "Self_Employed", "Dependents", "Credit_History"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())

    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = {}
    for col in cat_cols:
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col])

    # Train model
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    model = RandomForestClassifier()
    model.fit(X, y)

    return model, le

# Load model and encoders
model, le = load_model()

st.title("🏦 Loan Default Prediction")
st.markdown("Predict whether a loan will be approved and understand the reasons.")

# User inputs
input_data = {}

input_data["Gender"] = st.selectbox("Gender", ["Male", "Female"])
input_data["Married"] = st.selectbox("Married", ["Yes", "No"])
input_data["Dependents"] = st.selectbox("Dependents", ["0", "1", "2", "3+"])
input_data["Education"] = st.selectbox("Education", ["Graduate", "Not Graduate"])
input_data["Self_Employed"] = st.selectbox("Self Employed", ["Yes", "No"])
input_data["ApplicantIncome"] = st.number_input("Applicant Income", value=5000)
input_data["CoapplicantIncome"] = st.number_input("Coapplicant Income", value=0)
input_data["LoanAmount"] = st.number_input("Loan Amount (in thousands)", value=100)
input_data["Loan_Amount_Term"] = st.number_input("Loan Amount Term", value=360)
input_data["Credit_History"] = st.selectbox("Credit History", [1.0, 0.0])
input_data["Property_Area"] = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Encode input using trained encoders
    cat_cols = input_df.select_dtypes(include='object').columns.tolist()

    for col in cat_cols:
        if col in le and input_df[col][0] in le[col].classes_:
            input_df[col] = le[col].transform(input_df[col])
        else:
            st.error(f"Unknown category '{input_df[col][0]}' in column '{col}'.")
            st.stop()

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan is likely to be APPROVED")
    else:
        st.error("❌ Loan is likely to be REJECTED")

        # Reason analysis (simple logic)
        issues = []
        if input_data['Credit_History'] == 0.0:
            issues.append("Low Credit History")
        if input_data['ApplicantIncome'] < 3000:
            issues.append("Low Applicant Income")
        if input_data['LoanAmount'] > 200:
            issues.append("High Loan Amount")
        if input_data['Loan_Amount_Term'] < 180:
            issues.append("Short Repayment Term")

        if issues:
            st.markdown("**Possible reasons:**")
            for reason in issues:
                st.markdown(f"- {reason}")
        else:
            st.markdown("Could not determine specific reason.")

    st.markdown(f"### 🔍 Prediction Probability: {round(proba[prediction]*100, 2)}%")
