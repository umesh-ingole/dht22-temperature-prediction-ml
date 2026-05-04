import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data_3hrs.csv", encoding="latin1")

df.columns = df.columns.str.strip()

# ----------------------------
# Feature Engineering
# ----------------------------

# Convert timestamp
df['Timestamp (ms)'] = pd.to_datetime(df['Timestamp (ms)'], unit='ms')

# Time features
df['hour'] = df['Timestamp (ms)'].dt.hour
df['minute'] = df['Timestamp (ms)'].dt.minute
df['second'] = df['Timestamp (ms)'].dt.second

# Lag features
df['temp_lag1'] = df['Temperature (°C)'].shift(1)
df['temp_lag2'] = df['Temperature (°C)'].shift(2)
df['hum_lag1'] = df['Humidity (%)'].shift(1)

# Rolling features
df['temp_roll_mean_3'] = df['Temperature (°C)'].rolling(3).mean()
df['hum_roll_mean_3'] = df['Humidity (%)'].rolling(3).mean()

# Drop NaN rows
df = df.dropna().copy()

# ----------------------------
# Features & Target
# ----------------------------
X = df[[
    'Humidity (%)',
    'hour', 'minute', 'second',
    'temp_lag1', 'temp_lag2',
    'hum_lag1',
    'temp_roll_mean_3', 'hum_roll_mean_3'
]]

y = df['Temperature (°C)']

# ----------------------------
# Train-test split (NO SHUFFLE)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# ----------------------------
# XGBoost Model
# ----------------------------
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# ----------------------------
# Save model
# ----------------------------
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("XGBoost model trained successfully.")