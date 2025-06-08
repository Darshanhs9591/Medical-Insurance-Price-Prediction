import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Load model components (you'll need to save these first)
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('best_insurance_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, scaler, encoders
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please train the model first!")
        return None, None, None

# Preprocessing function
def preprocess_input(age, sex, bmi, children, smoker, region, scaler, encoders):
    """Preprocess user input to match training format"""
    try:
        # Encode categorical variables
        sex_encoded = encoders['sex'].transform([sex])[0]
        smoker_encoded = encoders['smoker'].transform([smoker])[0]
        region_encoded = encoders['region'].transform([region])[0]
        
        # Create interaction features
        age_bmi_interaction = age * bmi
        bmi_smoker_interaction = bmi * smoker_encoded
        age_smoker_interaction = age * smoker_encoded
        
        # Categorize BMI
        if bmi < 18.5:
            bmi_category = 0  # Underweight
        elif bmi < 25:
            bmi_category = 1  # Normal
        elif bmi < 30:
            bmi_category = 2  # Overweight
        else:
            bmi_category = 3  # Obese
        
        # Categorize age
        if age < 25:
            age_group = 0  # Young
        elif age < 40:
            age_group = 1  # Middle-aged
        elif age < 55:
            age_group = 2  # Mature
        else:
            age_group = 3  # Senior
        
        # Create feature array
        features = np.array([[
            age, bmi, children, sex_encoded, smoker_encoded, region_encoded,
            age_bmi_interaction, bmi_smoker_interaction, age_smoker_interaction,
            bmi_category, age_group
        ]])
        
        return features
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# Risk assessment function
def assess_risk(prediction):
    """Assess risk level based on prediction"""
    if prediction < 10000:
        return "Low Risk 🟢", "risk-low", "Excellent! You qualify for our lowest premium rates."
    elif prediction < 25000:
        return "Medium Risk 🟡", "risk-medium", "Moderate risk profile with competitive rates."
    else:
        return "High Risk 🔴", "risk-high", "Higher risk profile - consider lifestyle changes."

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Get instant insurance premium estimates based on your health profile</p>', unsafe_allow_html=True)
    
    # Load model components
    model, scaler, encoders = load_model_components()
    
    if model is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">📝 Personal Information</h2>', unsafe_allow_html=True)

# Input fields
    age = st.sidebar.slider("👤 Age", min_value=18, max_value=100, value=30, help="Your current age")

    sex = st.sidebar.selectbox("⚧ Gender", options=['male', 'female'], index=0, help="Select your gender")

    bmi = st.sidebar.number_input("⚖️ BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, 
                              help="Your BMI (weight in kg / height in m²)")

    children = st.sidebar.selectbox("👶 Number of Children", options=[0, 1, 2, 3, 4, 5], index=0, 
                                help="Number of children covered by insurance")

    smoker = st.sidebar.selectbox("🚬 Smoking Status", options=['no', 'yes'], index=0, help="Do you smoke?")

    region = st.sidebar.selectbox("🌍 Region", options=['southwest', 'southeast', 'northwest', 'northeast'], 
                              index=0, help="Your geographical region")

    
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🔮 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🚀 Predict Insurance Cost", type="primary", use_container_width=True):
            # Preprocess input
            features = preprocess_input(age, sex, bmi, children, smoker, region, scaler, encoders)
            
            if features is not None:
                # Make prediction
                prediction = model.predict(features)[0]
                risk_level, risk_class, risk_message = assess_risk(prediction)
                
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
                
                # Additional insights
                st.markdown("### 📊 Cost Breakdown Analysis")
                
                # Create comparison scenarios
                scenarios = {
                    "Your Profile": prediction,
                    "Non-Smoker (if applicable)": None,
                    "Normal BMI": None,
                    "Younger Age": None
                }
                
                # Calculate alternative scenarios
                if smoker == 'yes':
                    alt_features = preprocess_input(age, sex, bmi, children, 'no', region, scaler, encoders)
                    scenarios["Non-Smoker"] = model.predict(alt_features)[0]
                
                if bmi > 25:
                    alt_features = preprocess_input(age, sex, 24.9, children, smoker, region, scaler, encoders)
                    scenarios["Normal BMI (24.9)"] = model.predict(alt_features)[0]
                
                if age > 30:
                    alt_features = preprocess_input(25, sex, bmi, children, smoker, region, scaler, encoders)
                    scenarios["Age 25"] = model.predict(alt_features)[0]
                
                # Display scenarios
                scenario_data = []
                for scenario, cost in scenarios.items():
                    if cost is not None:
                        savings = prediction - cost if cost < prediction else 0
                        scenario_data.append({
                            "Scenario": scenario,
                            "Annual Cost": f"${cost:,.2f}",
                            "Potential Savings": f"${savings:,.2f}" if savings > 0 else "N/A"
                        })
                
                if scenario_data:
                    st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">📈 Your Profile</h2>', unsafe_allow_html=True)
        
        # Profile summary
        profile_data = {
            "Attribute": ["Age", "Gender", "BMI", "Children", "Smoker", "Region"],
            "Value": [age, sex.title(), f"{bmi:.1f}", children, smoker.title(), region.title()]
        }
        
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)
        
        # Health tips
        st.markdown("### 💡 Health Tips")
        tips = []
        
        if smoker == 'yes':
            tips.append("🚭 Quitting smoking can significantly reduce your premiums")
        
        if bmi > 30:
            tips.append("🏃‍♂️ Maintaining a healthy BMI can lower costs")
        
        if age > 50:
            tips.append("🏥 Regular health checkups become more important")
        
        tips.append("🥗 Healthy lifestyle choices pay off in lower premiums")
        tips.append("👨‍⚕️ Consider preventive healthcare measures")
        
        for tip in tips:
            st.info(tip)
    
    # Educational section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📚 Understanding Insurance Costs</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Key Factors:**
        - Smoking status (biggest impact)
        - Age and BMI
        - Number of dependents
        - Geographic location
        """)
    
    with col2:
        st.markdown("""
        **💰 Cost Ranges:**
        - Low Risk: Under $10,000
        - Medium Risk: $10,000 - $25,000
        - High Risk: Over $25,000
        """)
    
    with col3:
        st.markdown("""
        **🔧 Model Info:**
        - Algorithm: Gradient Boosting
        - Accuracy: 87%+ R² Score
        - Features: 11 engineered variables
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏥 Medical Insurance Cost Predictor | Built with Streamlit & Machine Learning</p>
        <p><em>Disclaimer: This is a predictive model for educational purposes. Actual insurance costs may vary.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
