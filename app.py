import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('xgb_model.pkl')

st.title("Customer Churn Prediction 🏃‍♂️‍➡️")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender ♂️ ️♀️", ["Male", "Female"])
    geography = st.selectbox("Geography 🗺️", ["Germany", "Spain"])
    credit_score = st.number_input("Credit Score ⭐", min_value=300, max_value=900)
    age = st.number_input("Age 🧑‍🦰", min_value=21, max_value=100)
    estimated_salary = st.number_input("Estimated Salary 💵")


with col2:
    balance = st.number_input("Account Balance 💰")
    num_of_products = st.selectbox("Number of Products 🧺", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card 💳", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member ⚡", ["Yes", "No"])
    tenure = st.slider("Tenure (Years with bank) ⏲️", 0, 10)


# Encode categorical values
gender = 1 if gender == "Male" else 0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# One-Hot Encode Geography (drop_first=True was likely used in training)
geography_Germany = 1 if geography == "Germany" else 0
geography_Spain = 1 if geography == "Spain" else 0
# France is the base case (drop_first=True), so no column for it

input_data = pd.DataFrame([[
    credit_score, gender, age, tenure, balance,
    num_of_products, has_cr_card, is_active_member,
    estimated_salary, geography_Germany, geography_Spain
]], columns=[
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("The customer is likely to churn. ❌")
    else:
        st.success("The customer is likely to stay. ✅")
