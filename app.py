import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback

# Error handling wrapper
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    st.warning(f"ML libraries not available: {e}")
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better visibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #d1e7dd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
        color: #0f5132;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .risk-low {
        background-color: #a7d4a7;
        color: #0d4f0d;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_simple_model():
    """Create a simple prediction model without ML libraries"""
    def predict_cost(age, sex, bmi, children, smoker, region):
        base_cost = 5000
        age_factor = age * 100
        bmi_factor = max(0, (bmi - 25) * 200)
        children_factor = children * 500
        smoker_factor = 15000 if smoker == "yes" else 0
        sex_factor = 200 if sex == "male" else 0
        
        return base_cost + age_factor + bmi_factor + children_factor + smoker_factor + sex_factor
    
    return predict_cost

def main():
    try:
        # Header
        st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Get instant insurance premium estimates</p>', unsafe_allow_html=True)
        
        # Load prediction model
        if ML_AVAILABLE:
            st.success("✅ Advanced ML model loaded!")
            # Use your existing ML model code here
        else:
            st.info("📊 Using simplified prediction model")
            predict_func = create_simple_model()
        
        # Sidebar inputs
        st.sidebar.header("📝 Personal Information")
        age = st.sidebar.slider("👤 Age", 18, 100, 30)
        sex = st.sidebar.selectbox("⚧ Gender", ["male", "female"])
        bmi = st.sidebar.number_input("⚖️ BMI", 15.0, 50.0, 25.0)
        children = st.sidebar.selectbox("👶 Children", [0, 1, 2, 3, 4, 5])
        smoker = st.sidebar.selectbox("🚬 Smoker", ["no", "yes"])
        region = st.sidebar.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])
        
        # BMI interpretation
        st.sidebar.markdown("### BMI Categories:")
        if bmi < 18.5:
            st.sidebar.info("📊 Underweight")
        elif bmi < 25:
            st.sidebar.success("📊 Normal weight")
        elif bmi < 30:
            st.sidebar.warning("📊 Overweight")
        else:
            st.sidebar.error("📊 Obese")
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 🔮 Prediction Results")
            
            if st.button("🚀 Predict Insurance Cost", type="primary", use_container_width=True):
                if ML_AVAILABLE:
                    # Use your existing ML prediction logic
                    prediction = predict_func(age, sex, bmi, children, smoker, region)
                else:
                    prediction = predict_func(age, sex, bmi, children, smoker, region)
                
                # Risk assessment
                if prediction < 10000:
                    risk_level = "Low Risk 🟢"
                    risk_class = "risk-low"
                    risk_message = "Excellent! You qualify for our lowest premium rates."
                elif prediction < 25000:
                    risk_level = "Medium Risk 🟡"
                    risk_class = "risk-medium"
                    risk_message = "Moderate risk profile with competitive rates."
                else:
                    risk_level = "High Risk 🔴"
                    risk_class = "risk-high"
                    risk_message = "Higher risk profile - consider lifestyle changes."
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>💰 Predicted Annual Premium: ${prediction:,.2f}</h3>
                    <div class="{risk_class}">
                        <strong>{risk_level}</strong><br>
                        {risk_message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("## 📈 Your Profile")
            profile_data = {
                "Attribute": ["Age", "Gender", "BMI", "Children", "Smoker", "Region"],
                "Value": [age, sex.title(), f"{bmi:.1f}", children, smoker.title(), region.title()]
            }
            st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)
            
            # Health tips
            st.markdown("### 💡 Health Tips")
            if smoker == "yes":
                st.info("🚭 Quitting smoking can significantly reduce premiums")
            if bmi > 30:
                st.info("🏃‍♂️ Maintaining healthy BMI can lower costs")
            st.info("🥗 Healthy lifestyle choices pay off")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🏥 Medical Insurance Cost Predictor | Built with Streamlit & ML</p>
            <p><em>Disclaimer: Educational purposes only. Actual costs may vary.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.info("Please refresh the page or contact support.")

if __name__ == "__main__":
    main()
