# Road Accident Risk Predictor 🚗

A machine learning-powered web application that predicts road accident risk based on various environmental and road conditions using Random Forest Regression.

## 🚀 Live Demo

Deploy this app on Streamlit Cloud: [https://share.streamlit.io](https://share.streamlit.io)

## 📊 Model Performance

- **Algorithm**: Random Forest Regressor
- **Accuracy**: 98% (R² = 0.9806)
- **Training Data**: 517,754 samples
- **Features**: 12 input features

## 🛠️ Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
CI_Project/
├── app.py                 # Streamlit web application
├── train.csv             # Training dataset
├── test.csv              # Test dataset
├── requirements.txt      # Dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── notebooks/
    └── road_accident_prediction.ipynb
```

## 🎯 Features

- Interactive web interface
- Real-time accident risk predictions
- Risk level visualization (Low/Medium/High)
- Responsive design

## 📈 Risk Levels

- 🟢 **Low Risk**: 0.0 - 0.3
- 🟡 **Medium Risk**: 0.3 - 0.6  
- 🔴 **High Risk**: 0.6 - 1.0

## 🚀 Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with `app.py` as main file

## 📋 Input Features

- Road Type (urban/highway/rural)
- Number of Lanes (1-4)
- Road Curvature (0-1)
- Speed Limit (25-70 mph)
- Lighting Conditions
- Weather Conditions
- Road Signs Present
- Public Road
- Time of Day
- Holiday Status
- School Season
- Number of Reported Accidents