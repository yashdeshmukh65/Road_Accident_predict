import streamlit as st
import pandas as pd
import numpy as np
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

@st.cache_resource
def get_model():
    """Create and return trained model"""
    try:
        df_train = pd.read_csv("data/train.csv")
        df_train.drop(columns=["id"], inplace=True)
        
        # Encode categorical columns
        categorical_mappings = {
            'road_type': {'highway': 0, 'rural': 1, 'urban': 2},
            'lighting': {'daylight': 0, 'dim': 1, 'night': 2},
            'weather': {'clear': 0, 'foggy': 1, 'rainy': 2},
            'time_of_day': {'afternoon': 0, 'evening': 1, 'morning': 2}
        }
        
        for col, mapping in categorical_mappings.items():
            df_train[col] = df_train[col].map(mapping)
        
        # Convert boolean columns
        bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
        for col in bool_cols:
            df_train[col] = df_train[col].astype(int)
        
        # Scale speed_limit
        df_train['speed_limit'] = (df_train['speed_limit'] - 25) / 45
        
        # Train model
        X = df_train.drop(columns=['accident_risk'])
        y = df_train['accident_risk']
        
        model = RandomForestRegressor(random_state=42, n_estimators=50)
        model.fit(X, y)
        
        return model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_input(data):
    """Preprocess input data"""
    df = data.copy()
    
    # Encode categorical variables
    encodings = {
        'road_type': {'highway': 0, 'rural': 1, 'urban': 2},
        'lighting': {'daylight': 0, 'dim': 1, 'night': 2},
        'weather': {'clear': 0, 'foggy': 1, 'rainy': 2},
        'time_of_day': {'afternoon': 0, 'evening': 1, 'morning': 2}
    }
    
    for col, mapping in encodings.items():
        df[col] = df[col].map(mapping)
    
    # Convert boolean columns
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # Scale speed_limit
    df['speed_limit'] = (df['speed_limit'] - 25) / 45
    
    return df

# Load model
model = get_model()

if model is not None:
    # Input form
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
            # Create input dataframe
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
                # Preprocess and predict
                processed_data = preprocess_input(input_data)
                prediction = model.predict(processed_data)[0]
                
                # Display results
                st.markdown("## 📊 Prediction Result")
                
                if prediction < 0.3:
                    st.success(f"🟢 **Low Risk**: {prediction:.4f}")
                elif prediction < 0.6:
                    st.warning(f"🟡 **Medium Risk**: {prediction:.4f}")
                else:
                    st.error(f"🔴 **High Risk**: {prediction:.4f}")
                
                # Display input summary
                st.markdown("### 📋 Input Summary")
                st.dataframe(input_data, use_container_width=True)

# Sidebar
st.sidebar.markdown("## ℹ️ About")
st.sidebar.markdown("""
This application predicts road accident risk using Random Forest Regression.

**Risk Levels:**
- 🟢 Low: 0.0 - 0.3
- 🟡 Medium: 0.3 - 0.6  
- 🔴 High: 0.6 - 1.0
""")