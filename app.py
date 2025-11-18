import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and subtitle
st.title("🚗 Road Accident Risk Predictor")
st.markdown("### Predict accident risk based on road conditions and environmental factors")

@st.cache_data
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure model.pkl exists in the project directory.")
        return None

def preprocess_data(df):
    """Preprocess the data similar to training"""
    df_processed = df.copy()
    
    # Define categorical columns
    categorical_columns = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                          'public_road', 'time_of_day', 'holiday', 'school_season']
    
    # Apply Label Encoding to categorical columns
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # Apply MinMax scaling to speed_limit
    if 'speed_limit' in df_processed.columns:
        scaler = MinMaxScaler()
        df_processed['speed_limit'] = scaler.fit_transform(df_processed[['speed_limit']])
    
    return df_processed

def make_predictions(model, data):
    """Make predictions using the loaded model"""
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

# Load model
model = load_model()

if model is not None:
    # Input form for prediction
    st.markdown("## 📝 Enter Road Condition Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            road_type = st.selectbox("Road Type", ["urban", "highway", "rural"])
            num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=4, value=2)
            curvature = st.slider("Road Curvature", 0.0, 1.0, 0.5, 0.01)
            speed_limit = st.selectbox("Speed Limit", [25, 35, 45, 60, 70])
            lighting = st.selectbox("Lighting Conditions", ["daylight", "dim", "night"])
            weather = st.selectbox("Weather Conditions", ["clear", "rainy", "foggy"])
        
        with col2:
            road_signs_present = st.checkbox("Road Signs Present", value=True)
            public_road = st.checkbox("Public Road", value=True)
            time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening"])
            holiday = st.checkbox("Holiday")
            school_season = st.checkbox("School Season")
            num_reported_accidents = st.number_input("Number of Reported Accidents", min_value=0, max_value=10, value=1)
        
        submitted = st.form_submit_button("🔮 Predict Accident Risk", type="primary")
        
        if submitted:
            # Create dataframe from inputs
            input_data = pd.DataFrame({
                'road_type': [road_type],
                'num_lanes': [num_lanes],
                'curvature': [curvature],
                'speed_limit': [speed_limit],
                'lighting': [lighting],
                'weather': [weather],
                'road_signs_present': [road_signs_present],
                'public_road': [public_road],
                'time_of_day': [time_of_day],
                'holiday': [holiday],
                'school_season': [school_season],
                'num_reported_accidents': [num_reported_accidents]
            })
            
            with st.spinner("Making prediction..."):
                # Preprocess data
                processed_data = preprocess_data(input_data)
                
                # Make prediction
                prediction = make_predictions(model, processed_data)
                
                if prediction is not None:
                    # Display results
                    st.markdown("## 📊 Prediction Result")
                    
                    risk_score = prediction[0]
                    
                    # Display risk score with color coding
                    if risk_score < 0.3:
                        st.success(f"🟢 **Low Risk**: {risk_score:.4f}")
                    elif risk_score < 0.6:
                        st.warning(f"🟡 **Medium Risk**: {risk_score:.4f}")
                    else:
                        st.error(f"🔴 **High Risk**: {risk_score:.4f}")
                    
                    # Display input summary
                    st.markdown("### 📋 Input Summary")
                    st.dataframe(input_data, use_container_width=True)

else:
    st.error("❌ Unable to load the model. Please check if model.pkl exists in the project directory.")
    st.markdown("### 🔧 Model Training Required")
    st.markdown("Please run the notebook in the `notebooks/` folder to train and save the model first.")

# Sidebar with additional information
st.sidebar.markdown("## ℹ️ About")
st.sidebar.markdown("""
This application predicts road accident risk based on various factors including:
- Road characteristics
- Environmental conditions  
- Traffic conditions
- Temporal factors

The model uses Random Forest Regression trained on historical accident data.
""")

st.sidebar.markdown("## 🛠️ Model Info")
st.sidebar.markdown("""
- **Algorithm**: Random Forest Regressor
- **Features**: 12 input features
- **Output**: Accident risk score (0-1)
- **Training Data**: 517,754 samples
""")