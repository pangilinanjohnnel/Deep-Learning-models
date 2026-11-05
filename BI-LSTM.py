import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score

# Load datasets
exo = pd.read_csv(r"archive\MJ_Days_Played_Exog_Data.csv")
nike = pd.read_csv(r"archive\Nike_Historical_Stock_Price_Refined.csv")

# Merge on Date
merge = pd.merge(nike, exo, on="Date")

# Set index and sort
merge['Date'] = pd.to_datetime(merge['Date'])
merge.set_index('Date', inplace=True)
merge.sort_index(inplace=True)

# Fill missing values
clean = merge.asfreq('B').ffill()

X_data = clean[["Adj Close", "Exogeneous_Variable"]]
y_data = clean[["Adj Close"]]

# 1. Define the split point (80%)
split_idx = int(len(X_data) * 0.8)

# 2. Scale them based ONLY on the first 80%
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# ONLY 'fit' on the training portion (the past)
scaler_X.fit(X_data.iloc[:split_idx])
scaler_y.fit(y_data.iloc[:split_idx])

# Now 'transform' the whole thing using those training rules
X_scaled = scaler_X.transform(X_data)
y_scaled = scaler_y.transform(y_data)

window_size = 30

X = []
y = []

for i in range(window_size, len(X_scaled)):
    # X takes the previous 30 days of BOTH features
    X.append(X_scaled[i - window_size:i])

    # y takes ONLY the price at the current day
    y.append(y_scaled[i])

X = np.array(X)
y = np.array(y)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(Bidirectional(LSTM(50), input_shape=(window_size, 2)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test))

pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(pred)
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

#plot
plt.plot(y_actual, label="Actual", c='blue')
plt.plot(y_pred, label="Predicted", c='red')
plt.title("Nike Stock Price Prediction (BI-LSTM Model)")
plt.xlabel("Days in Test Set")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()


#metrics
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_actual , y_pred)
mape = mean_absolute_percentage_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
print(f"BI-LSTM:MSE:{mse:.4f},RMSE:{rmse:.4f},MAE:{mae:.4f}, MAPE: {mape*100:.4f}%,R2:{r2:.4f}")
