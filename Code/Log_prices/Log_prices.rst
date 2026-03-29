.. code:: ipython3

    # SECTION 1
    # !pip install yfinance pmdarima tensorflow scikit-learn matplotlib pandas numpy
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt
    
    import pmdarima as pm
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
    from tensorflow.keras.callbacks import EarlyStopping

.. code:: ipython3

    # SECTION 2
    data = yf.download("^GSPC", start="2006-01-01", end="2026-01-01")
    
    # Convert to Series with proper index
    prices = data[['Close']].copy()
    prices.columns = ['price']
    prices = prices['price']
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    
    print(prices.head())
    print("Length:", len(prices))


.. parsed-literal::

    [*********************100%***********************]  1 of 1 completed

.. parsed-literal::

    Date
    2006-01-03    1268.800049
    2006-01-04    1273.459961
    2006-01-05    1273.479980
    2006-01-06    1285.449951
    2006-01-09    1290.150024
    Name: price, dtype: float64
    Length: 5031
    

.. parsed-literal::

    
    

.. code:: ipython3

    # SECTION 3
    log_prices = np.log(prices)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns.values.ravel()  # 1D for ARIMA

.. code:: ipython3

    # SECTION 3A
    # Moving averages
    ma20 = prices.rolling(window=20).mean()
    ma50 = prices.rolling(window=50).mean()
    
    # Volatility
    volatility = prices.rolling(window=20).std()
    
    # RSI
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Combine features
    features = pd.DataFrame({
        'log_price': log_prices,
        'ma20': np.log(ma20),
        'ma50': np.log(ma50),
        'volatility': volatility,
        'rsi': rsi
    }).dropna()
    
    print(features.head())


.. parsed-literal::

                log_price      ma20      ma50  volatility        rsi
    Date                                                            
    2006-03-15   7.172440  7.159402  7.154134    7.625111  58.922089
    2006-03-16   7.174211  7.160386  7.154705    8.593533  59.242004
    2006-03-17   7.175681  7.161080  7.155233    9.670885  57.885900
    2006-03-20   7.174020  7.161772  7.155726   10.365653  66.967715
    2006-03-21   7.167986  7.162322  7.155910   10.407531  54.326153
    

.. code:: ipython3

    # SECTION 4
    split = int(len(log_prices) * 0.8)
    
    train_prices = log_prices[:split]
    test_prices = log_prices[split:]
    
    train_returns = log_returns[:split]
    test_returns = log_returns[split:]

.. code:: ipython3

    # SECTION 5
    arima_model = pm.auto_arima(train_returns, seasonal=False, stepwise=True)
    n_periods = len(test_returns)
    arima_forecast_returns = arima_model.predict(n_periods=n_periods)
    
    last_train_log_price = train_prices.iloc[-1]
    arima_forecast_log_prices = np.cumsum(arima_forecast_returns) + last_train_log_price
    arima_forecast_prices = np.exp(arima_forecast_log_prices)
    
    arima_series = pd.Series(
        arima_forecast_prices,
        index=prices.index[-len(arima_forecast_prices):]
    )
    print(arima_series.head())


.. parsed-literal::

    Date
    2021-12-29    4793.678745
    2021-12-30    4795.124963
    2021-12-31    4796.720161
    2022-01-03    4798.294749
    2022-01-04    4799.872862
    dtype: float64
    

.. code:: ipython3

    # SECTION 6
    SEQ_LEN = 120
    
    # Separate scaler for log_price target
    price_scaler = MinMaxScaler()
    log_price_scaled = price_scaler.fit_transform(features[['log_price']])
    
    # Scale all features for input
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    # Replace log_price column with scaled target (optional, keeps sequence consistent)
    scaled_features[:,0] = log_price_scaled.ravel()
    
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i,0])  # predict log_price only
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_features, SEQ_LEN)
    
    split_seq = int(len(X) * 0.8)
    X_train, X_test = X[:split_seq], X[split_seq:]
    y_train, y_test = y[:split_seq], y[split_seq:]
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)


.. parsed-literal::

    X_train shape: (3889, 120, 5)
    y_train shape: (3889,)
    

.. code:: ipython3

    # SECTION 7
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    
    lstm_model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs=50,
                   batch_size=32,
                   callbacks=[early_stop],
                   verbose=1)


.. parsed-literal::

    Epoch 1/50
    

.. parsed-literal::

    C:\Users\DevAdmin\AppData\Roaming\Python\Python311\site-packages\keras\src\layers\rnn\rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(**kwargs)
    

.. parsed-literal::

    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m16s[0m 97ms/step - loss: 0.0062 - val_loss: 1.6896e-04
    Epoch 2/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 94ms/step - loss: 0.0018 - val_loss: 1.3127e-04
    Epoch 3/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 87ms/step - loss: 0.0016 - val_loss: 3.6979e-04
    Epoch 4/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 92ms/step - loss: 0.0013 - val_loss: 1.6111e-04
    Epoch 5/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 90ms/step - loss: 0.0013 - val_loss: 1.8206e-04
    Epoch 6/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 88ms/step - loss: 0.0011 - val_loss: 2.3922e-04
    Epoch 7/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 85ms/step - loss: 0.0012 - val_loss: 5.2121e-04
    



.. parsed-literal::

    <keras.src.callbacks.history.History at 0x279d1c4a450>



.. code:: ipython3

    # SECTION 7
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    
    lstm_model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs=50,
                   batch_size=32,
                   callbacks=[early_stop],
                   verbose=1)


.. parsed-literal::

    Epoch 1/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 83ms/step - loss: 0.0086 - val_loss: 0.0021
    Epoch 2/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 84ms/step - loss: 0.0017 - val_loss: 4.2181e-04
    Epoch 3/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 81ms/step - loss: 0.0015 - val_loss: 7.8792e-04
    Epoch 4/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 85ms/step - loss: 0.0013 - val_loss: 3.4142e-04
    Epoch 5/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 83ms/step - loss: 0.0012 - val_loss: 1.6908e-04
    Epoch 6/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 85ms/step - loss: 0.0014 - val_loss: 4.2081e-04
    Epoch 7/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 84ms/step - loss: 0.0013 - val_loss: 2.7240e-04
    Epoch 8/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 86ms/step - loss: 0.0012 - val_loss: 0.0015
    Epoch 9/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 85ms/step - loss: 0.0011 - val_loss: 1.9390e-04
    Epoch 10/50
    [1m122/122[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 86ms/step - loss: 9.9136e-04 - val_loss: 3.3928e-04
    



.. parsed-literal::

    <keras.src.callbacks.history.History at 0x279d8317e10>



.. code:: ipython3

    # SECTION 9
    lstm_preds = lstm_model.predict(X_test)
    cnn_lstm_preds = cnn_lstm_model.predict(X_test)
    
    # Only inverse-transform log_price (first column)
    lstm_preds_price = price_scaler.inverse_transform(lstm_preds)
    cnn_lstm_preds_price = price_scaler.inverse_transform(cnn_lstm_preds)
    y_test_price = price_scaler.inverse_transform(y_test.reshape(-1,1))
    
    # Convert log_price → actual prices
    lstm_prices = np.exp(lstm_preds_price)
    cnn_lstm_prices = np.exp(cnn_lstm_preds_price)
    actual_prices = np.exp(y_test_price)
    
    print("Predictions ready")


.. parsed-literal::

    [1m31/31[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 37ms/step
    [1m31/31[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    Predictions ready
    

.. code:: ipython3

    # SECTION 10
    min_len = min(len(actual_prices), len(arima_series), len(lstm_prices), len(cnn_lstm_prices))
    
    actual_prices = actual_prices[-min_len:]
    lstm_prices = lstm_prices[-min_len:]
    cnn_lstm_prices = cnn_lstm_prices[-min_len:]
    arima_series = arima_series[-min_len:]
    
    dates = prices.index[-min_len:]

.. code:: ipython3

    # SECTION 11
    plt.figure(figsize=(14,7))
    plt.plot(dates, actual_prices, label="Actual")
    plt.plot(dates, arima_series, label="ARIMA")
    plt.plot(dates, lstm_prices, label="LSTM")
    plt.plot(dates, cnn_lstm_prices, label="CNN-LSTM")
    plt.title("S&P 500 Forecast Comparison with Technical Indicators")
    plt.legend()
    plt.show()



.. image:: output_11_0.png


.. code:: ipython3

    # SECTION 12
    actual_1d = np.ravel(actual_prices)
    arima_1d = np.ravel(arima_series)
    lstm_1d = np.ravel(lstm_prices)
    cnn_lstm_1d = np.ravel(cnn_lstm_prices)
    
    def rmse(a,b): return sqrt(mean_squared_error(a,b))
    def mae(a,b): return mean_absolute_error(a,b)
    def mape(a,b): return np.mean(np.abs((a-b)/a))*100
    
    results = pd.DataFrame({
        "Model": ["ARIMA","LSTM","CNN-LSTM"],
        "RMSE": [
            rmse(actual_1d, arima_1d),
            rmse(actual_1d, lstm_1d),
            rmse(actual_1d, cnn_lstm_1d)
        ],
        "MAE": [
            mae(actual_1d, arima_1d),
            mae(actual_1d, lstm_1d),
            mae(actual_1d, cnn_lstm_1d)
        ],
        "MAPE": [
            mape(actual_1d, arima_1d),
            mape(actual_1d, lstm_1d),
            mape(actual_1d, cnn_lstm_1d)
        ]
    })
    
    results




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Model</th>
          <th>RMSE</th>
          <th>MAE</th>
          <th>MAPE</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ARIMA</td>
          <td>836.737398</td>
          <td>715.632156</td>
          <td>16.233608</td>
        </tr>
        <tr>
          <th>1</th>
          <td>LSTM</td>
          <td>143.651460</td>
          <td>108.156386</td>
          <td>2.237397</td>
        </tr>
        <tr>
          <th>2</th>
          <td>CNN-LSTM</td>
          <td>283.621012</td>
          <td>227.944147</td>
          <td>4.226009</td>
        </tr>
      </tbody>
    </table>
    </div>



