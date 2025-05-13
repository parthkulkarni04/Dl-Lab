import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
from math import sqrt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Download Apple Stock Price Data
df = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
data = df[['Close']]
data.dropna(inplace=True)
print("Data shape:", data.shape)
print(data.head())

# 2. Normalize Data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3. Create Sequences
def create_sequences(data, seq_length=50):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# 4. Train-Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build LSTM Model
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=False, input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# 6. Train the Model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# 7. Predict
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# 8. Evaluate
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# 9. Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM: Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid()
plt.show()

# 10. Confusion Matrix (Using Discretization)
def categorize(values, bins=10):
    return pd.qcut(values.flatten(), bins, labels=False)

y_test_cat = categorize(y_test_inv)
y_pred_cat = categorize(y_pred_inv)

cm = confusion_matrix(y_test_cat, y_pred_cat)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Discretized)")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.show()

--------------------------
##LSTM Weather prediction 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from meteostat import Point, Daily

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
from math import sqrt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Get Weather Data for Delhi using meteostat
start = datetime(2020, 1, 1)
end = datetime(2023, 1, 1)

# Delhi coordinates
delhi = Point(28.6139, 77.2090)

# Fetch daily weather data
data = Daily(delhi, start, end)
data = data.fetch()
print(data.head())

# Use average daily temperature in Celsius
data = data[['tavg']].dropna()
data.rename(columns={'tavg': 'temp'}, inplace=True)

# 2. Normalize Temperature Data
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(data)

# 3. Create Sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(temp_scaled, SEQ_LENGTH)

# 4. Train-Test Split (80% training)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build LSTM Model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# 6. Train Model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# 7. Predict & Inverse Transform
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# 8. Evaluate Performance
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# 9. Plot Actual vs Predicted Temperatures
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual Temperature (°C)')
plt.plot(y_pred_inv, label='Predicted Temperature (°C)')
plt.title('Delhi Daily Temperature Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.show()

# 10. Confusion Matrix (Quantized)
def categorize(values, bins=10):
    return pd.qcut(values.flatten(), bins, labels=False)

y_test_cat = categorize(y_test_inv)
y_pred_cat = categorize(y_pred_inv)

cm = confusion_matrix(y_test_cat, y_pred_cat)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix (Binned Temperature Categories)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
