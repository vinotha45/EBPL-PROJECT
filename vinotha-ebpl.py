import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --- 1. Download stock data ---
def get_stock_data(ticker='AAPL', start='2015-01-01', end='2023-12-31'):
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

# --- 2. Prepare data for LSTM ---
def prepare_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        x_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

# --- 3. Build LSTM Model ---
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# --- 4. Predict future stock prices ---
def predict_future(model, data, scaler, window_size=60):
    last_60_days = data[-window_size:].values
    scaled_last_60 = scaler.transform(last_60_days)

    X_test = []
    X_test.append(scaled_last_60)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price[0][0]

# --- Main Function ---
if __name__ == '__main__':
    df = get_stock_data('AAPL')
    df.plot(title="AAPL Stock Price")
    plt.show()

    x_train, y_train, scaler = prepare_data(df)

    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    future_price = predict_future(model, df[['Close']], scaler)
    print(f"Predicted next closing price: ${future_price:.2f}")
