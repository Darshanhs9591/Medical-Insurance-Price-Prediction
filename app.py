import streamlit as st
import pandas as pd
import numpy as np

# Basic page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Simple header
st.title("🏥 Medical Insurance Cost Predictor")
st.write("✅ App is running successfully!")

# Basic form
st.sidebar.header("📝 Personal Information")
age = st.sidebar.slider("👤 Age", 18, 100, 30)
sex = st.sidebar.selectbox("⚧ Gender", ["male", "female"])
bmi = st.sidebar.number_input("⚖️ BMI", 15.0, 50.0, 25.0)
children = st.sidebar.selectbox("👶 Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("🚬 Smoker", ["no", "yes"])
region = st.sidebar.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])

# Simple prediction logic
if st.button("🚀 Predict Insurance Cost", type="primary"):
    # Basic calculation without ML libraries
    base_cost = 5000
    age_factor = age * 100
    bmi_factor = max(0, (bmi - 25) * 200)
    children_factor = children * 500
    smoker_factor = 15000 if smoker == "yes" else 0
    
    prediction = base_cost + age_factor + bmi_factor + children_factor + smoker_factor
    
    st.success(f"💰 Predicted Annual Premium: ${prediction:,.2f}")
    
    if prediction < 10000:
        st.info("🟢 Low Risk - Excellent rates!")
    elif prediction < 25000:
        st.warning("🟡 Medium Risk - Competitive rates")
    else:
        st.error("🔴 High Risk - Consider lifestyle changes")

st.write("🎉 Deployment successful! App is working.")
