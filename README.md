# Road Accident Risk Predictor

A machine learning-powered web application that predicts road accident risk based on various environmental and road conditions.

## 📁 Project Structure

```
project/
├── app.py                          # Main Streamlit application
├── model.pkl                       # Trained machine learning model
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   ├── train.csv                   # Training dataset
│   └── test.csv                    # Test dataset
└── notebooks/
    └── road_accident_prediction.ipynb  # Model training notebook
```

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit interface
- **Interactive Form**: Enter road conditions through user-friendly inputs
- **Real-time Predictions**: Get instant accident risk predictions
- **Results Visualization**: View prediction statistics and distributions
- **CSV Download**: Download prediction results as CSV files
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Local Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd CI_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if model.pkl doesn't exist)
   - Open `notebooks/road_accident_prediction.ipynb`
   - Run all cells to train and save the model
   - The model will be saved as `model.pkl` in the project root

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 🌐 Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository
1. Push your code to GitHub
2. Ensure all files are in the repository:
   - `app.py`
   - `requirements.txt`
   - `model.pkl` (train the model first)
   - `data/` folder with CSV files
   - `notebooks/` folder with the notebook

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to `app.py`
6. Click "Deploy"

### Step 3: Access Your App
- Your app will be available at: `https://your-app-name.streamlit.app`
- Share the URL with others to use your app

## 📊 Model Information

- **Algorithm**: Random Forest Regressor
- **Training Data**: 517,754 samples
- **Features**: 12 input features
- **Performance**: 
  - R² Score: 0.98 (on training data)
  - MSE: 0.0005 (on training data)

### Input Features
1. `road_type`: Type of road (urban, highway, rural)
2. `num_lanes`: Number of lanes
3. `curvature`: Road curvature (0-1)
4. `speed_limit`: Speed limit
5. `lighting`: Lighting conditions (daylight, dim, night)
6. `weather`: Weather conditions (clear, rainy, foggy)
7. `road_signs_present`: Whether road signs are present (True/False)
8. `public_road`: Whether it's a public road (True/False)
9. `time_of_day`: Time of day (morning, afternoon, evening)
10. `holiday`: Whether it's a holiday (True/False)
11. `school_season`: Whether it's school season (True/False)
12. `num_reported_accidents`: Number of reported accidents

### Output
- `accident_risk`: Predicted accident risk score (0-1, where 1 is highest risk)

## 📝 Usage Instructions

### Using the Web App

1. **Enter Road Data**:
   - Fill in all the required fields in the form
   - Select appropriate values from dropdowns
   - Adjust sliders and checkboxes as needed

2. **Make Prediction**:
   - Click "Predict Accident Risk" button
   - View the risk score and interpretation

3. **Interpret Results**:
   - 🟢 Low Risk: 0.0 - 0.3
   - 🟡 Medium Risk: 0.3 - 0.6
   - 🔴 High Risk: 0.6 - 1.0

## 🔧 Model Training Code

Add this code to your notebook to save the trained model:

```python
import joblib
from sklearn.ensemble import RandomForestRegressor

# Train your model (assuming you have X_train, y_train)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
```

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `model.pkl` exists in the project root
   - Run the notebook to train and save the model

2. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

3. **CSV upload issues**
   - Ensure CSV has all required columns
   - Check for proper column names and data types

4. **Streamlit not starting**
   - Check if port 8501 is available
   - Try: `streamlit run app.py --server.port 8502`

### Getting Help

- Check the Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Review the notebook for model training details
- Ensure all dependencies are properly installed

## 📈 Future Enhancements

- [ ] Add model performance metrics display
- [ ] Implement data validation and error handling
- [ ] Add more visualization options
- [ ] Support for different model algorithms
- [ ] Batch prediction capabilities
- [ ] API endpoint for programmatic access

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.