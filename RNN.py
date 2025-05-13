import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix
from math import sqrt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Load Stock Price Data (e.g., AAPL)
df = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
data = df[['Close']]  # Use only the closing price
data.dropna(inplace=True)
print("Data shape:", data.shape)
print(data.head())

# 2. Normalize Data (0 to 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3. Create Sequences (time-steps)
def create_sequences(data, seq_length=50):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# 4. Train-Test Split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build Simple RNN Model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 6. Train Model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# 7. Predict and Inverse Transform
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# 8. Evaluate with MSE and RMSE
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# 9. Plot: Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid()
plt.show()

# 10. Confusion Matrix (For Regression - Discretized Buckets)
def categorize(values, bins=10):
    """Discretize continuous values into bins for confusion matrix"""
    return pd.qcut(values.flatten(), bins, labels=False)

y_test_cat = categorize(y_test_inv)
y_pred_cat = categorize(y_pred_inv)

cm = confusion_matrix(y_test_cat, y_pred_cat)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Binned Categories)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
