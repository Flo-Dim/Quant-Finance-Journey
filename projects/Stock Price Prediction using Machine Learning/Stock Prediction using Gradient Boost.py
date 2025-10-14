import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
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

# From earlier: X_train, X_test, y_train, y_test

split_val = int(0.9 * len(X_train))
X_tr, X_val = X_train.iloc[:split_val], X_train.iloc[split_val:]
y_tr, y_val = y_train.iloc[:split_val], y_train.iloc[split_val:]
X_tr.shape, X_val.shape

# Lets train the XGBoost 

xgb = XGBRegressor(
    n_estimators=1000,          # upper bound; early stopping will stop earlier
    learning_rate=0.05,         # small steps
    max_depth=6,                # how complex each tree is
    subsample=0.8,              # use 80% of rows per tree (reduces overfit)
    colsample_bytree=0.8,       # use 80% of features per tree
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse',
    early_stopping_rounds=50    # stop if no improvement for 50 rounds
)

xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=False, 
)
# Make predictions
y_pred_xgb = xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost - MAE: {mae_xgb:.4f} | RMSE: {rmse_xgb:.4f}")

# Compare predicted values and real values 
pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb}).head(10)
