import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Download the data
ticker = "AAPL"
period = "7y"      
interval = "1d"

raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
raw = raw[['Open','High','Low','Close','Volume']].dropna().sort_index()

# Features on X
df = raw.copy()
df['Close_lag1'] = df['Close'].shift(1)     # yesterday's price 
df['Close_lag2'] = df['Close'].shift(2)     # 2 days ago's price
df['SMA_5']      = df['Close'].rolling(5).mean()  # Fast moving Average
df['SMA_10']     = df['Close'].rolling(10).mean()  # Slow Moving Average

# Target value Y (Tomorrow's price)
df['y_next_close'] = df['Close'].shift(-1) 
df_model = df.dropna().copy()
# df_model.head()

# Split values into clues and target
feature_cols = ['Close_lag1','Close_lag2','SMA_5','SMA_10','Volume']
X = df_model[feature_cols]
y = df_model['y_next_close']

print("Rows available:", X.shape[0])
X.head(3), y.head(3)

# Split into train and test set,  Ratio (80:20)
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Let's scale our features based on training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVR model
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# Make predictions 
y_pred_svr = svr.predict(X_test_scaled)

# Evaluate our model's performance 
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = sqrt(mean_squared_error(y_test, y_pred_svr))
print(f"SVR - MAE: {mae_svr:.4f} | RMSE: {rmse_svr:.4f}")

# Compare the first ten (10) predictions to real values 
compare_svr = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_svr
}).head(10)
compare_svr