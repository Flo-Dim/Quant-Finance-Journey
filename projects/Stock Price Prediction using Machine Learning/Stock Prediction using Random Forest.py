import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
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

# Let's initialize the model

rf = RandomForestRegressor(
    n_estimators=200,   # number of trees
    max_depth=10,       # how deep each tree can grow
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

# Train our model on raw data
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest - MAE: {mae_rf:.4f} | RMSE: {rmse_rf:.4f}")

# Compare the first ten (10) predictions to real values 
compare_rf = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf
}).head(10)
compare_rf