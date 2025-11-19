import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
df_train = pd.read_csv("data/train.csv")
df_train.drop(columns=["id"], inplace=True)

# Get categorical columns
categorical_columns = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()

# Apply Label Encoding
for col in categorical_columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])

# Apply MinMax scaling to speed_limit
scaler = MinMaxScaler()
df_train['speed_limit'] = scaler.fit_transform(df_train[['speed_limit']])

# Prepare features and target
X_train = df_train.drop(columns=['accident_risk'])
y_train = df_train['accident_risk']

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'model.pkl')
print("✅ Model saved successfully as model.pkl")
print(f"Model accuracy (R²): {rf_model.score(X_train, y_train):.4f}")