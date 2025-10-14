# import libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Download the data
ticker = "AAPL"
period = "7y"      
interval = "1d"

raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
raw = raw[['Open','High','Low','Close','Volume']].dropna().sort_index()

print(raw.head(3))
print(raw.tail(3))
print(raw.shape)

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

# initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Display the evalution metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Compare the first five (5) predictions to real values 
compare = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).head(5)
compare
