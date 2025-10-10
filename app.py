import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .prediction-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_data
def load_model():
    try:
        model = joblib.load('best_diabetes_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found! Please run the notebook first to generate the model files.")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Predict diabetes risk based on health parameters")
    
    # Load model
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📊 Health Parameters")
    st.sidebar.markdown("Enter the following health parameters to predict diabetes risk:")
    
    # Input fields
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1, help="Number of pregnancies")
    glucose = st.sidebar.slider("Glucose", 0, 200, 100, help="Glucose concentration (mg/dL)")
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70, help="Diastolic blood pressure (mm Hg)")
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20, help="Triceps skin fold thickness (mm)")
    insulin = st.sidebar.slider("Insulin", 0, 846, 50, help="2-Hour serum insulin (mu U/ml)")
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0, help="Body mass index (kg/m²)")
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5, help="Diabetes pedigree function")
    age = st.sidebar.slider("Age", 21, 81, 30, help="Age in years")
    
    # Create input data
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Display input summary
    st.subheader("📋 Input Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.write(f"• Age: {age} years")
        st.write(f"• Pregnancies: {pregnancies}")
        st.write(f"• BMI: {bmi:.1f} kg/m²")
    
    with col2:
        st.markdown("**Medical Values:**")
        st.write(f"• Glucose: {glucose} mg/dL")
        st.write(f"• Blood Pressure: {blood_pressure} mm Hg")
        st.write(f"• Insulin: {insulin} mu U/ml")
    
    # Prediction button
    if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Preprocess the data (same as training)
        # Replace 0s with NaN for imputation
        input_data_processed = input_data.copy()
        input_data_processed.replace(0, np.nan, inplace=True)
        
        # Scale the data
        input_scaled = scaler.transform(input_data_processed)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_scaled_df)
        prediction_proba = model.predict_proba(input_scaled_df) if hasattr(model, 'predict_proba') else None
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        if prediction[0] == 1:
            st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
            st.markdown("### ⚠️ HIGH RISK")
            st.markdown("**The model predicts a HIGH risk of diabetes.**")
            st.markdown("**Recommendation:** Please consult with a healthcare professional for further evaluation and monitoring.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
            st.markdown("### ✅ LOW RISK")
            st.markdown("**The model predicts a LOW risk of diabetes.**")
            st.markdown("**Recommendation:** Continue maintaining a healthy lifestyle with regular check-ups.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show confidence if available
        if prediction_proba is not None:
            confidence = max(prediction_proba[0]) * 100
            st.metric("Prediction Confidence", f"{confidence:.1f}%")
        
        # Show cluster information
        st.info(f"**Risk Category:** Cluster {prediction[0]} (0=Low Risk, 1=High Risk)")
    
    # Information section
    st.subheader("ℹ️ About This Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Information:**")
        st.write("• Trained on diabetes dataset")
        st.write("• Uses machine learning clustering")
        st.write("• Optimized with hyperparameter tuning")
        st.write("• High accuracy prediction model")
    
    with col2:
        st.markdown("**Disclaimer:**")
        st.write("• This is a prediction tool only")
        st.write("• Not a substitute for medical advice")
        st.write("• Consult healthcare professionals")
        st.write("• For educational purposes")
    
    # Footer
    st.markdown("---")
    st.markdown("**Developed with ❤️ using Streamlit and Scikit-learn**")

if __name__ == "__main__":
    main()
