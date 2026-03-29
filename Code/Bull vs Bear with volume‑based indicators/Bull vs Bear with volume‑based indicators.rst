.. code:: ipython3

    !pip install tabulate --user


.. parsed-literal::

    Collecting tabulate
      Downloading tabulate-0.10.0-py3-none-any.whl.metadata (40 kB)
    Downloading tabulate-0.10.0-py3-none-any.whl (39 kB)
    Installing collected packages: tabulate
    Successfully installed tabulate-0.10.0
    

.. code:: ipython3

    !pip install --upgrade jinja2 --user
    


.. parsed-literal::

    Requirement already satisfied: jinja2 in C:\Users\DevAdmin\AppData\Roaming\Python\Python311\site-packages (3.1.5)
    Collecting jinja2
      Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
    Requirement already satisfied: MarkupSafe>=2.0 in C:\Users\DevAdmin\AppData\Roaming\Python\Python311\site-packages (from jinja2) (2.1.5)
    Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
    Installing collected packages: jinja2
      Attempting uninstall: jinja2
        Found existing installation: Jinja2 3.1.5
        Uninstalling Jinja2-3.1.5:
          Successfully uninstalled Jinja2-3.1.5
    Successfully installed jinja2-3.1.6
    

.. code:: ipython3

    # =========================================
    # 1. Imports
    # =========================================
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    

.. code:: ipython3

    # =========================================
    # 2. Config
    # =========================================
    
    TICKER = "^GSPC"
    START_DATE = "2000-01-01"
    END_DATE = "2024-01-01"
    
    LOOKBACK = 60
    TEST_SIZE_RATIO = 0.2
    VAL_SIZE_RATIO = 0.1
    
    BATCH_SIZE = 64
    EPOCHS = 30
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    

.. code:: ipython3

    # =========================================
    # 3. Download data
    # =========================================
    
    import yfinance as yf
    
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df.head()
    


.. parsed-literal::

    [*********************100%***********************]  1 of 1 completed
    



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
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2000-01-03</th>
          <td>1469.250000</td>
          <td>1478.000000</td>
          <td>1438.359985</td>
          <td>1455.219971</td>
          <td>931800000</td>
        </tr>
        <tr>
          <th>2000-01-04</th>
          <td>1455.219971</td>
          <td>1455.219971</td>
          <td>1397.430054</td>
          <td>1399.420044</td>
          <td>1009000000</td>
        </tr>
        <tr>
          <th>2000-01-05</th>
          <td>1399.420044</td>
          <td>1413.270020</td>
          <td>1377.680054</td>
          <td>1402.109985</td>
          <td>1085500000</td>
        </tr>
        <tr>
          <th>2000-01-06</th>
          <td>1402.109985</td>
          <td>1411.900024</td>
          <td>1392.099976</td>
          <td>1403.449951</td>
          <td>1092300000</td>
        </tr>
        <tr>
          <th>2000-01-07</th>
          <td>1403.449951</td>
          <td>1441.469971</td>
          <td>1400.729980</td>
          <td>1441.469971</td>
          <td>1225200000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # =========================================
    # 4. Base features
    # =========================================
    
    df['ret_1d'] = df['Close'].pct_change()
    df['log_volume'] = np.log(df['Volume'] + 1)
    
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    
    df['vol_20'] = df['ret_1d'].rolling(20).std()
    

.. code:: ipython3

    # =========================================
    # 5. Volume-based indicators (amplified)
    # =========================================
    
    # OBV
    price_change = df['Close'].diff()
    direction = np.sign(price_change).fillna(0)
    df['obv'] = (direction * df['Volume']).cumsum()
    
    # Volume Rate of Change (VROC)
    df['vroc_10'] = df['Volume'].pct_change(10)
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    tp_diff = typical_price.diff()
    pos_mf = raw_money_flow.where(tp_diff > 0, 0).rolling(14).sum()
    neg_mf = raw_money_flow.where(tp_diff < 0, 0).rolling(14).sum().abs()
    
    mfr = pos_mf / (neg_mf + 1e-9)
    df['mfi_14'] = 100 - (100 / (1 + mfr))
    
    # Chaikin Money Flow (CMF)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
    mf_volume = mf_multiplier * df['Volume']
    df['cmf_20'] = mf_volume.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-9)
    
    # Accumulation/Distribution Line (ADL)
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
    df['adl'] = (clv * df['Volume']).cumsum()
    
    df = df.dropna()
    df.head()
    




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
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume</th>
          <th>ret_1d</th>
          <th>log_volume</th>
          <th>sma_20</th>
          <th>sma_50</th>
          <th>sma_200</th>
          <th>vol_20</th>
          <th>obv</th>
          <th>vroc_10</th>
          <th>mfi_14</th>
          <th>cmf_20</th>
          <th>adl</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2000-10-16</th>
          <td>1374.170044</td>
          <td>1379.479980</td>
          <td>1365.060059</td>
          <td>1374.619995</td>
          <td>1005400000</td>
          <td>0.000327</td>
          <td>20.728651</td>
          <td>1418.555499</td>
          <td>1462.176399</td>
          <td>1444.783248</td>
          <td>0.013517</td>
          <td>-7.720860e+09</td>
          <td>-0.043569</td>
          <td>28.811270</td>
          <td>-0.052714</td>
          <td>-2.146472e+09</td>
        </tr>
        <tr>
          <th>2000-10-17</th>
          <td>1374.619995</td>
          <td>1380.989990</td>
          <td>1342.339966</td>
          <td>1349.969971</td>
          <td>1161500000</td>
          <td>-0.017932</td>
          <td>20.872978</td>
          <td>1413.058997</td>
          <td>1459.589399</td>
          <td>1444.256998</td>
          <td>0.013576</td>
          <td>-8.882360e+09</td>
          <td>0.057736</td>
          <td>28.940084</td>
          <td>-0.121884</td>
          <td>-2.849382e+09</td>
        </tr>
        <tr>
          <th>2000-10-18</th>
          <td>1349.969971</td>
          <td>1356.650024</td>
          <td>1305.790039</td>
          <td>1342.130005</td>
          <td>1441700000</td>
          <td>-0.005808</td>
          <td>21.089089</td>
          <td>1407.598499</td>
          <td>1456.775999</td>
          <td>1443.970548</td>
          <td>0.013576</td>
          <td>-1.032406e+10</td>
          <td>0.234967</td>
          <td>20.971801</td>
          <td>-0.111453</td>
          <td>-2.230864e+09</td>
        </tr>
        <tr>
          <th>2000-10-19</th>
          <td>1342.130005</td>
          <td>1389.930054</td>
          <td>1342.130005</td>
          <td>1388.760010</td>
          <td>1297900000</td>
          <td>0.034743</td>
          <td>20.984013</td>
          <td>1404.583997</td>
          <td>1455.093799</td>
          <td>1443.903798</td>
          <td>0.016088</td>
          <td>-9.026160e+09</td>
          <td>0.103563</td>
          <td>28.800666</td>
          <td>-0.083401</td>
          <td>-9.965040e+08</td>
        </tr>
        <tr>
          <th>2000-10-20</th>
          <td>1388.760010</td>
          <td>1408.469971</td>
          <td>1382.189941</td>
          <td>1396.930054</td>
          <td>1177400000</td>
          <td>0.005883</td>
          <td>20.886574</td>
          <td>1401.994501</td>
          <td>1453.827400</td>
          <td>1443.871199</td>
          <td>0.016181</td>
          <td>-7.848760e+09</td>
          <td>0.023737</td>
          <td>35.854115</td>
          <td>-0.127194</td>
          <td>-8.531286e+08</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # =========================================
    # 6. Regime labels (Bull = 1, Bear = 0)
    # =========================================
    
    df['regime'] = (df['Close'] > df['sma_200']).astype(int)
    
    df['regime'].value_counts()
    




.. parsed-literal::

    regime
    1    4052
    0    1786
    Name: count, dtype: int64



.. code:: ipython3

    # =========================================
    # 7. Build sequences
    # =========================================
    
    feature_cols = [
        'Open','High','Low','Close','Volume',
        'ret_1d','log_volume',
        'sma_20','sma_50','sma_200',
        'vol_20',
        'obv','vroc_10','mfi_14','cmf_20','adl'
    ]
    
    X_all = df[feature_cols].values
    y_all = df['regime'].values
    
    X_seq, y_seq = [], []
    
    for i in range(LOOKBACK, len(df)):
        X_seq.append(X_all[i-LOOKBACK:i])
        y_seq.append(y_all[i])
    
    X = np.array(X_seq)
    y = np.array(y_seq)
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    


.. parsed-literal::

    X shape: (5778, 60, 16)
    y shape: (5778,)
    

.. code:: ipython3

    # =========================================
    # 8. Train/val/test split
    # =========================================
    
    n = len(X)
    test = int(n * TEST_SIZE_RATIO)
    train_val = n - test
    
    X_train_val = X[:train_val]
    y_train_val = y[:train_val]
    
    X_test = X[train_val:]
    y_test = y[train_val:]
    
    val = int(len(X_train_val) * VAL_SIZE_RATIO)
    
    X_train = X_train_val[:-val]
    y_train = y_train_val[:-val]
    
    X_val = X_train_val[-val:]
    y_val = y_train_val[-val:]
    
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)
    


.. parsed-literal::

    Train: (4161, 60, 16)
    Val: (462, 60, 16)
    Test: (1155, 60, 16)
    

.. code:: ipython3

    # =========================================
    # 9. CLEAN EXTREME VALUES + SCALING
    # =========================================
    
    # --- 1. Replace infinities with NaN ---
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # --- 2. Clip extreme indicator values (prevents scaler overflow) ---
    df['vroc_10'] = df['vroc_10'].clip(-10, 10)        # cap ±1000%
    df['cmf_20'] = df['cmf_20'].clip(-5, 5)
    df['mfi_14'] = df['mfi_14'].clip(0, 100)           # natural bounds
    df['obv'] = df['obv'].clip(-1e9, 1e9)
    df['adl'] = df['adl'].clip(-1e9, 1e9)
    
    # --- 3. Drop rows with NaN after cleaning ---
    df = df.dropna()
    
    # --- 4. Rebuild X after cleaning ---
    X_all = df[feature_cols].values
    y_all = df['regime'].values
    
    X_seq, y_seq = [], []
    for i in range(LOOKBACK, len(df)):
        X_seq.append(X_all[i-LOOKBACK:i])
        y_seq.append(y_all[i])
    
    X = np.array(X_seq)
    y = np.array(y_seq)
    
    # --- 5. Train/val/test split (unchanged) ---
    n = len(X)
    test = int(n * TEST_SIZE_RATIO)
    train_val = n - test
    
    X_train_val = X[:train_val]
    y_train_val = y[:train_val]
    
    X_test = X[train_val:]
    y_test = y[train_val:]
    
    val = int(len(X_train_val) * VAL_SIZE_RATIO)
    
    X_train = X_train_val[:-val]
    y_train = y_train_val[:-val]
    
    X_val = X_train_val[-val:]
    y_val = y_train_val[-val:]
    
    # --- 6. Scaling (now safe) ---
    n_features = X.shape[2]
    scaler = StandardScaler()
    
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    
    scaler.fit(X_train_2d)
    
    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)
    
    print("Scaling complete. Shapes:")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)
    


.. parsed-literal::

    Scaling complete. Shapes:
    Train: (4160, 60, 16)
    Val: (462, 60, 16)
    Test: (1155, 60, 16)
    

.. code:: ipython3

    # =========================================
    # 10. Class weights (balanced)
    # =========================================
    
    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight = {i: cw[i] for i in range(len(cw))}
    class_weight
    




.. parsed-literal::

    {0: 1.5441722345953972, 1: 0.7394241023817988}



.. code:: ipython3

    # =========================================
    # 11. LSTM model (binary)
    # =========================================
    
    def build_lstm(input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
    
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    
    lstm_model = build_lstm((LOOKBACK, n_features))
    lstm_model.summary()
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ lstm_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">20,736</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">22,849</span> (89.25 KB)
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">22,849</span> (89.25 KB)
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
    </pre>
    


.. code:: ipython3

    # =========================================
    # 12. Train LSTM
    # =========================================
    
    lstm_hist = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1
    )
    


.. parsed-literal::

    Epoch 1/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 29ms/step - accuracy: 0.8368 - loss: 0.3705 - val_accuracy: 0.8312 - val_loss: 0.4438
    Epoch 2/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9272 - loss: 0.1863 - val_accuracy: 0.8810 - val_loss: 0.2495
    Epoch 3/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - accuracy: 0.9368 - loss: 0.1555 - val_accuracy: 0.8961 - val_loss: 0.2501
    Epoch 4/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 23ms/step - accuracy: 0.9440 - loss: 0.1369 - val_accuracy: 0.8658 - val_loss: 0.4209
    Epoch 5/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 24ms/step - accuracy: 0.9524 - loss: 0.1173 - val_accuracy: 0.8571 - val_loss: 0.4599
    Epoch 6/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 26ms/step - accuracy: 0.9577 - loss: 0.1087 - val_accuracy: 0.8810 - val_loss: 0.3240
    Epoch 7/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 28ms/step - accuracy: 0.9575 - loss: 0.0987 - val_accuracy: 0.8831 - val_loss: 0.4090
    Epoch 8/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9630 - loss: 0.0929 - val_accuracy: 0.8853 - val_loss: 0.3397
    Epoch 9/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 25ms/step - accuracy: 0.9632 - loss: 0.0924 - val_accuracy: 0.8745 - val_loss: 0.4294
    Epoch 10/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 25ms/step - accuracy: 0.9651 - loss: 0.0878 - val_accuracy: 0.8918 - val_loss: 0.3715
    Epoch 11/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9688 - loss: 0.0821 - val_accuracy: 0.8939 - val_loss: 0.2558
    Epoch 12/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 24ms/step - accuracy: 0.9697 - loss: 0.0796 - val_accuracy: 0.8831 - val_loss: 0.3930
    Epoch 13/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9659 - loss: 0.0856 - val_accuracy: 0.8874 - val_loss: 0.3593
    Epoch 14/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 25ms/step - accuracy: 0.9721 - loss: 0.0720 - val_accuracy: 0.8528 - val_loss: 0.5533
    Epoch 15/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9673 - loss: 0.0772 - val_accuracy: 0.8918 - val_loss: 0.3547
    Epoch 16/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 26ms/step - accuracy: 0.9675 - loss: 0.0735 - val_accuracy: 0.8896 - val_loss: 0.3405
    Epoch 17/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 26ms/step - accuracy: 0.9685 - loss: 0.0759 - val_accuracy: 0.8939 - val_loss: 0.3509
    Epoch 18/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 24ms/step - accuracy: 0.9690 - loss: 0.0741 - val_accuracy: 0.8874 - val_loss: 0.3265
    Epoch 19/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 26ms/step - accuracy: 0.9714 - loss: 0.0730 - val_accuracy: 0.8874 - val_loss: 0.3216
    Epoch 20/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 26ms/step - accuracy: 0.9695 - loss: 0.0687 - val_accuracy: 0.8853 - val_loss: 0.3279
    Epoch 21/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9692 - loss: 0.0697 - val_accuracy: 0.8918 - val_loss: 0.3889
    Epoch 22/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9688 - loss: 0.0710 - val_accuracy: 0.8896 - val_loss: 0.3292
    Epoch 23/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 25ms/step - accuracy: 0.9680 - loss: 0.0674 - val_accuracy: 0.8896 - val_loss: 0.3834
    Epoch 24/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9707 - loss: 0.0679 - val_accuracy: 0.8874 - val_loss: 0.3264
    Epoch 25/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9712 - loss: 0.0637 - val_accuracy: 0.8874 - val_loss: 0.3401
    Epoch 26/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9675 - loss: 0.0805 - val_accuracy: 0.8874 - val_loss: 0.3407
    Epoch 27/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 25ms/step - accuracy: 0.9736 - loss: 0.0661 - val_accuracy: 0.8896 - val_loss: 0.3685
    Epoch 28/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9714 - loss: 0.0683 - val_accuracy: 0.8831 - val_loss: 0.4740
    Epoch 29/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9714 - loss: 0.0645 - val_accuracy: 0.8896 - val_loss: 0.3650
    Epoch 30/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - accuracy: 0.9716 - loss: 0.0636 - val_accuracy: 0.8918 - val_loss: 0.4179
    

.. code:: ipython3

    # =========================================
    # 13. CNN-LSTM model
    # =========================================
    
    def build_cnn_lstm(input_shape):
        inp = layers.Input(shape=input_shape)
    
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(inp)
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
    
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
    
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
    
        out = layers.Dense(1, activation='sigmoid')(x)
    
        model = models.Model(inp, out)
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    
    cnn_lstm_model = build_cnn_lstm((LOOKBACK, n_features))
    cnn_lstm_model.summary()
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_9"</span>
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ input_layer_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv1d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">1,568</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ conv1d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">3,104</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ max_pooling1d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ lstm_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">24,832</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">31,617</span> (123.50 KB)
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">31,617</span> (123.50 KB)
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
    </pre>
    


.. code:: ipython3

    # =========================================
    # 14. Train CNN-LSTM
    # =========================================
    
    cnn_hist = cnn_lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1
    )
    


.. parsed-literal::

    Epoch 1/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 23ms/step - accuracy: 0.8522 - loss: 0.3875 - val_accuracy: 0.8810 - val_loss: 0.3155
    Epoch 2/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.9212 - loss: 0.2162 - val_accuracy: 0.8442 - val_loss: 0.3907
    Epoch 3/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 24ms/step - accuracy: 0.9332 - loss: 0.1749 - val_accuracy: 0.8420 - val_loss: 0.4303
    Epoch 4/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.9380 - loss: 0.1520 - val_accuracy: 0.8701 - val_loss: 0.3680
    Epoch 5/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 17ms/step - accuracy: 0.9411 - loss: 0.1369 - val_accuracy: 0.8766 - val_loss: 0.3308
    Epoch 6/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9457 - loss: 0.1271 - val_accuracy: 0.9437 - val_loss: 0.1703
    Epoch 7/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - accuracy: 0.9495 - loss: 0.1187 - val_accuracy: 0.9416 - val_loss: 0.1770
    Epoch 8/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9514 - loss: 0.1107 - val_accuracy: 0.9199 - val_loss: 0.2407
    Epoch 9/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.9570 - loss: 0.1030 - val_accuracy: 0.9177 - val_loss: 0.2410
    Epoch 10/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9546 - loss: 0.1107 - val_accuracy: 0.9004 - val_loss: 0.2817
    Epoch 11/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9558 - loss: 0.0974 - val_accuracy: 0.8961 - val_loss: 0.3856
    Epoch 12/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.9577 - loss: 0.1052 - val_accuracy: 0.9264 - val_loss: 0.2133
    Epoch 13/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9603 - loss: 0.0925 - val_accuracy: 0.9004 - val_loss: 0.3198
    Epoch 14/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9623 - loss: 0.0954 - val_accuracy: 0.9026 - val_loss: 0.3174
    Epoch 15/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9673 - loss: 0.0864 - val_accuracy: 0.9307 - val_loss: 0.2183
    Epoch 16/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - accuracy: 0.9611 - loss: 0.0856 - val_accuracy: 0.9026 - val_loss: 0.3136
    Epoch 17/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9608 - loss: 0.0932 - val_accuracy: 0.9113 - val_loss: 0.2749
    Epoch 18/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.9613 - loss: 0.0989 - val_accuracy: 0.9113 - val_loss: 0.2603
    Epoch 19/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.9668 - loss: 0.0794 - val_accuracy: 0.9329 - val_loss: 0.1899
    Epoch 20/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9637 - loss: 0.0881 - val_accuracy: 0.9199 - val_loss: 0.2189
    Epoch 21/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.9697 - loss: 0.0781 - val_accuracy: 0.9264 - val_loss: 0.2374
    Epoch 22/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - accuracy: 0.9651 - loss: 0.0822 - val_accuracy: 0.9134 - val_loss: 0.3008
    Epoch 23/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 16ms/step - accuracy: 0.9663 - loss: 0.0808 - val_accuracy: 0.9372 - val_loss: 0.1740
    Epoch 24/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - accuracy: 0.9695 - loss: 0.0769 - val_accuracy: 0.9221 - val_loss: 0.2596
    Epoch 25/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - accuracy: 0.9663 - loss: 0.0769 - val_accuracy: 0.9091 - val_loss: 0.3413
    Epoch 26/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 21ms/step - accuracy: 0.9678 - loss: 0.0778 - val_accuracy: 0.9048 - val_loss: 0.3581
    Epoch 27/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 17ms/step - accuracy: 0.9680 - loss: 0.0725 - val_accuracy: 0.9481 - val_loss: 0.1318
    Epoch 28/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 18ms/step - accuracy: 0.9685 - loss: 0.0736 - val_accuracy: 0.9372 - val_loss: 0.1875
    Epoch 29/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - accuracy: 0.9685 - loss: 0.0693 - val_accuracy: 0.9026 - val_loss: 0.4217
    Epoch 30/30
    [1m65/65[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 19ms/step - accuracy: 0.9678 - loss: 0.0699 - val_accuracy: 0.9502 - val_loss: 0.0941
    

.. code:: ipython3

    # =========================================
    # 15. Evaluation
    # =========================================
    
    def evaluate(model, X, y_true, name):
        y_prob = model.predict(X).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
    
        print(f"\n=== {name} ===")
        print(classification_report(y_true, y_pred, digits=4))
        print(confusion_matrix(y_true, y_pred))
    
        return y_prob, y_pred
    
    lstm_prob, lstm_pred = evaluate(lstm_model, X_test, y_test, "LSTM (Bull/Bear)")
    cnn_prob, cnn_pred = evaluate(cnn_lstm_model, X_test, y_test, "CNN-LSTM (Bull/Bear)")
    


.. parsed-literal::

    [1m37/37[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 13ms/step
    
    === LSTM (Bull/Bear) ===
                  precision    recall  f1-score   support
    
               0     0.7156    0.2591    0.3805       301
               1     0.7868    0.9637    0.8663       854
    
        accuracy                         0.7801      1155
       macro avg     0.7512    0.6114    0.6234      1155
    weighted avg     0.7682    0.7801    0.7397      1155
    
    [[ 78 223]
     [ 31 823]]
    [1m37/37[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 12ms/step
    
    === CNN-LSTM (Bull/Bear) ===
                  precision    recall  f1-score   support
    
               0     0.8129    0.8804    0.8453       301
               1     0.9566    0.9286    0.9424       854
    
        accuracy                         0.9160      1155
       macro avg     0.8847    0.9045    0.8938      1155
    weighted avg     0.9191    0.9160    0.9171      1155
    
    [[265  36]
     [ 61 793]]
    

.. code:: ipython3

    # =========================================
    # 16. Simple regime-based strategy check
    # =========================================
    
    test_idx = df.index[-len(y_test):]
    test_close = df.loc[test_idx, 'Close'].values
    
    def regime_strategy_stats(pred, close, name):
        # Long only in Bull regime (1)
        bull_mask = (pred == 1)
        if bull_mask.sum() == 0:
            print(f"{name}: no bull periods detected.")
            return
    
        ret = np.diff(close) / close[:-1]
        ret = ret[-len(pred):]  # align lengths
    
        strat_ret = ret[bull_mask[:-1]]  # last pred has no next-day return
    
        print(f"\n{name} strategy:")
        print("  # bull days:", bull_mask.sum())
        print("  avg daily return in bull:", strat_ret.mean())
        print("  hit ratio (ret>0):", (strat_ret > 0).mean())
    
    regime_strategy_stats(lstm_pred, test_close, "LSTM Bull regime")
    regime_strategy_stats(cnn_pred, test_close, "CNN-LSTM Bull regime")
    


.. parsed-literal::

    
    LSTM Bull regime strategy:
      # bull days: 1046
      avg daily return in bull: 0.00054267680921807
      hit ratio (ret>0): 0.5358851674641149
    
    CNN-LSTM Bull regime strategy:
      # bull days: 829
      avg daily return in bull: 0.0005019792727858908
      hit ratio (ret>0): 0.5543478260869565
    

.. code:: ipython3

    # =========================================
    # FULL EVALUATION FOR LSTM + CNN-LSTM
    # =========================================
    
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score
    )
    
    def evaluate_model(model, X_test, y_test, name="Model"):
        print(f"\n===== {name} Evaluation =====")
    
        # Predict probabilities
        y_prob = model.predict(X_test).ravel()
    
        # Convert to class labels
        y_pred = (y_prob >= 0.5).astype(int)
    
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
    
        # Precision, Recall, F1
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
    
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))
    
        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC:   {auc:.4f}")
        except:
            print("ROC-AUC:   Could not compute")
    
        return y_prob, y_pred
    
    
    # ---- Run evaluation for BOTH models ----
    
    lstm_prob, lstm_pred = evaluate_model(
        lstm_model,
        X_test,
        y_test,
        name="LSTM (Bull/Bear)"
    )
    
    cnn_prob, cnn_pred = evaluate_model(
        cnn_lstm_model,
        X_test,
        y_test,
        name="CNN-LSTM (Bull/Bear)"
    )
    


.. parsed-literal::

    
    ===== LSTM (Bull/Bear) Evaluation =====
    [1m37/37[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    Accuracy: 0.7801
    Precision: 0.7868
    Recall:    0.9637
    F1 Score:  0.8663
    
    Confusion Matrix:
    [[ 78 223]
     [ 31 823]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0     0.7156    0.2591    0.3805       301
               1     0.7868    0.9637    0.8663       854
    
        accuracy                         0.7801      1155
       macro avg     0.7512    0.6114    0.6234      1155
    weighted avg     0.7682    0.7801    0.7397      1155
    
    ROC-AUC:   0.9042
    
    ===== CNN-LSTM (Bull/Bear) Evaluation =====
    [1m37/37[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step
    Accuracy: 0.9160
    Precision: 0.9566
    Recall:    0.9286
    F1 Score:  0.9424
    
    Confusion Matrix:
    [[265  36]
     [ 61 793]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0     0.8129    0.8804    0.8453       301
               1     0.9566    0.9286    0.9424       854
    
        accuracy                         0.9160      1155
       macro avg     0.8847    0.9045    0.8938      1155
    weighted avg     0.9191    0.9160    0.9171      1155
    
    ROC-AUC:   0.9733
    

.. code:: ipython3

    import pandas as pd
    from tabulate import tabulate
    
    table1 = pd.DataFrame({
        "Partition": ["Training Set", "Validation Set", "Test Set"],
        "Samples": [len(X_train), len(X_val), len(X_test)],
        "Percentage (%)": [
            round(len(X_train)/len(X)*100, 2),
            round(len(X_val)/len(X)*100, 2),
            round(len(X_test)/len(X)*100, 2)
        ]
    })
    
    print("\nTABLE 1: Dataset Partitioning Description")
    print(tabulate(table1, headers='keys', tablefmt='fancy_grid', showindex=False))
    


.. parsed-literal::

    
    TABLE 1: Dataset Partitioning Description
    ╒════════════════╤═══════════╤══════════════════╕
    │ Partition      │   Samples │   Percentage (%) │
    ╞════════════════╪═══════════╪══════════════════╡
    │ Training Set   │      4160 │            72.01 │
    ├────────────────┼───────────┼──────────────────┤
    │ Validation Set │       462 │             8    │
    ├────────────────┼───────────┼──────────────────┤
    │ Test Set       │      1155 │            19.99 │
    ╘════════════════╧═══════════╧══════════════════╛
    

.. code:: ipython3

    import pandas as pd
    
    def model_to_table_safe(model):
        layers_data = []
    
        for layer in model.layers:
            # Layer name
            name = layer.name
    
            # Layer type
            layer_type = layer.__class__.__name__
    
            # Output shape (safe extraction)
            try:
                out_shape = layer.output_shape
            except:
                try:
                    out_shape = layer.output.shape
                except:
                    out_shape = "N/A"
    
            # Parameter count
            try:
                params = layer.count_params()
            except:
                params = "N/A"
    
            layers_data.append([name, layer_type, str(out_shape), params])
    
        df = pd.DataFrame(layers_data, columns=["Layer Name", "Layer Type", "Output Shape", "Parameters"])
        return df
    
    
    table2 = model_to_table_safe(cnn_lstm_model)
    
    print("\nTABLE 2: Summary of the Proposed CNN-LSTM Architecture")
    
    
    print(tabulate(table2, headers='keys', tablefmt='fancy_grid', showindex=False))
    


.. parsed-literal::

    
    TABLE 2: Summary of the Proposed CNN-LSTM Architecture
    ╒═════════════════╤══════════════╤════════════════╤══════════════╕
    │ Layer Name      │ Layer Type   │ Output Shape   │   Parameters │
    ╞═════════════════╪══════════════╪════════════════╪══════════════╡
    │ input_layer_5   │ InputLayer   │ (None, 60, 16) │            0 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ conv1d_4        │ Conv1D       │ (None, 60, 32) │         1568 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ conv1d_5        │ Conv1D       │ (None, 60, 32) │         3104 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ max_pooling1d_2 │ MaxPooling1D │ (None, 30, 32) │            0 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ dropout_12      │ Dropout      │ (None, 30, 32) │            0 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ lstm_5          │ LSTM         │ (None, 64)     │        24832 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ dropout_13      │ Dropout      │ (None, 64)     │            0 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ dense_10        │ Dense        │ (None, 32)     │         2080 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ dropout_14      │ Dropout      │ (None, 32)     │            0 │
    ├─────────────────┼──────────────┼────────────────┼──────────────┤
    │ dense_11        │ Dense        │ (None, 1)      │           33 │
    ╘═════════════════╧══════════════╧════════════════╧══════════════╛
    

.. code:: ipython3

    table3 = pd.DataFrame({
        "Model": ["LSTM (Baseline)", "CNN-LSTM (Proposed)"],
        "Accuracy (%)": [
            round(accuracy_score(y_test, lstm_pred)*100, 2),
            round(accuracy_score(y_test, cnn_pred)*100, 2)
        ]
    })
    
    print("\nTABLE 3: Comparison of Proposed System With Existing Approaches")
    print(tabulate(table3, headers='keys', tablefmt='fancy_grid', showindex=False))
    


.. parsed-literal::

    
    TABLE 3: Comparison of Proposed System With Existing Approaches
    ╒═════════════════════╤════════════════╕
    │ Model               │   Accuracy (%) │
    ╞═════════════════════╪════════════════╡
    │ LSTM (Baseline)     │          78.01 │
    ├─────────────────────┼────────────────┤
    │ CNN-LSTM (Proposed) │          91.6  │
    ╘═════════════════════╧════════════════╛
    

.. code:: ipython3

    print(table1.to_latex(index=False, caption="Dataset Partitioning Description"))

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(confusion_matrix(y_test, lstm_pred), annot=True, fmt="d",
                cmap="Blues", ax=axes[0])
    axes[0].set_title("LSTM Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    sns.heatmap(confusion_matrix(y_test, cnn_pred), annot=True, fmt="d",
                cmap="Greens", ax=axes[1])
    axes[1].set_title("CNN-LSTM Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    



.. image:: output_23_0.png


.. code:: ipython3

    plt.figure(figsize=(14,6))
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(lstm_hist.history['accuracy'], label='Train Accuracy')
    plt.plot(lstm_hist.history['val_accuracy'], label='Val Accuracy')
    plt.title("LSTM Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(lstm_hist.history['loss'], label='Train Loss')
    plt.plot(lstm_hist.history['val_loss'], label='Val Loss')
    plt.title("LSTM Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    



.. image:: output_24_0.png


.. code:: ipython3

    plt.figure(figsize=(14,6))
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(cnn_hist.history['accuracy'], label='Train Accuracy')
    plt.plot(cnn_hist.history['val_accuracy'], label='Val Accuracy')
    plt.title("CNN-LSTM Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(cnn_hist.history['loss'], label='Train Loss')
    plt.plot(cnn_hist.history['val_loss'], label='Val Loss')
    plt.title("CNN-LSTM Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    



.. image:: output_25_0.png


.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    
    # ---- LSTM metrics ----
    cm_lstm = confusion_matrix(y_test, lstm_pred)
    
    lstm_accuracy = accuracy_score(y_test, lstm_pred)
    lstm_sensitivity = cm_lstm[1,1] / (cm_lstm[1,1] + cm_lstm[1,0])     # Recall for class 1
    lstm_specificity = cm_lstm[0,0] / (cm_lstm[0,0] + cm_lstm[0,1])     # Recall for class 0
    lstm_f1 = f1_score(y_test, lstm_pred)
    
    # ---- CNN-LSTM metrics ----
    cm_cnn = confusion_matrix(y_test, cnn_pred)
    
    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    cnn_sensitivity = cm_cnn[1,1] / (cm_cnn[1,1] + cm_cnn[1,0])
    cnn_specificity = cm_cnn[0,0] / (cm_cnn[0,0] + cm_cnn[0,1])
    cnn_f1 = f1_score(y_test, cnn_pred)
    
    # ---- Prepare data ----
    metrics = ["Accuracy", "Specificity", "Sensitivity", "F1-Score"]
    
    lstm_values = [lstm_accuracy, lstm_specificity, lstm_sensitivity, lstm_f1]
    cnn_values  = [cnn_accuracy, cnn_specificity, cnn_sensitivity, cnn_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # ---- Plot ----
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, lstm_values, width, label="LSTM", color="#1f77b4")
    plt.bar(x + width/2, cnn_values, width, label="CNN-LSTM", color="#2ca02c")
    
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Performance Comparison of LSTM vs CNN-LSTM")
    
    # ⭐ Add legend
    plt.legend(loc="upper left")
    
    # Optional: add grid for cleaner academic look
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    



.. image:: output_26_0.png


.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    
    # ---- LSTM metrics ----
    cm_lstm = confusion_matrix(y_test, lstm_pred)
    
    lstm_accuracy = accuracy_score(y_test, lstm_pred)
    lstm_precision = precision_score(y_test, lstm_pred)
    lstm_recall = recall_score(y_test, lstm_pred)  # Sensitivity
    lstm_specificity = cm_lstm[0,0] / (cm_lstm[0,0] + cm_lstm[0,1])
    lstm_f1 = f1_score(y_test, lstm_pred)
    
    # ---- CNN-LSTM metrics ----
    cm_cnn = confusion_matrix(y_test, cnn_pred)
    
    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    cnn_precision = precision_score(y_test, cnn_pred)
    cnn_recall = recall_score(y_test, cnn_pred)
    cnn_specificity = cm_cnn[0,0] / (cm_cnn[0,0] + cm_cnn[0,1])
    cnn_f1 = f1_score(y_test, cnn_pred)
    
    # ---- Prepare data ----
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    
    lstm_values = [lstm_accuracy, lstm_precision, lstm_recall, lstm_specificity, lstm_f1]
    cnn_values  = [cnn_accuracy, cnn_precision, cnn_recall, cnn_specificity, cnn_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # ---- Plot ----
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, lstm_values, width, label="LSTM", color="#1f77b4")
    plt.bar(x + width/2, cnn_values, width, label="CNN-LSTM", color="#2ca02c")
    
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Performance Metrics Comparison: LSTM vs CNN-LSTM")
    
    plt.ylim(0, 1.1)
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    



.. image:: output_27_0.png


.. code:: ipython3

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # ---- Compute ROC curves ----
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_prob)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
    
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_prob)
    roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
    
    # ---- Plot ----
    plt.figure(figsize=(8,6))
    
    plt.plot(fpr_lstm, tpr_lstm, color="blue",
             label=f"LSTM (AUC = {roc_auc_lstm:.3f})", linewidth=2)
    
    plt.plot(fpr_cnn, tpr_cnn, color="green",
             label=f"CNN-LSTM (AUC = {roc_auc_cnn:.3f})", linewidth=2)
    
    # Diagonal reference line
    plt.plot([0,1], [0,1], 'k--', linewidth=1)
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: LSTM vs CNN-LSTM")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.show()
    



.. image:: output_28_0.png


.. code:: ipython3

    import pandas as pd
    from sklearn.metrics import matthews_corrcoef
    
    # Compute MCC for both models
    mcc_lstm = matthews_corrcoef(y_test, lstm_pred)
    mcc_cnn  = matthews_corrcoef(y_test, cnn_pred)
    
    # Create a table
    mcc_table = pd.DataFrame({
        "Model": ["LSTM", "CNN-LSTM"],
        "MCC": [mcc_lstm, mcc_cnn]
    })
    
    # Display as a clean Markdown table
    print("\nMCC Comparison Table:\n")
    print(tabulate(mcc_table, headers='keys', tablefmt='fancy_grid', showindex=False))


.. parsed-literal::

    
    MCC Comparison Table:
    
    ╒══════════╤══════════╕
    │ Model    │      MCC │
    ╞══════════╪══════════╡
    │ LSTM     │ 0.334595 │
    ├──────────┼──────────┤
    │ CNN-LSTM │ 0.788967 │
    ╘══════════╧══════════╛
    

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, f1_score,
        precision_score, recall_score, matthews_corrcoef
    )
    
    # ---- LSTM metrics ----
    cm_lstm = confusion_matrix(y_test, lstm_pred)
    
    lstm_accuracy = accuracy_score(y_test, lstm_pred)
    lstm_precision = precision_score(y_test, lstm_pred)
    lstm_recall = recall_score(y_test, lstm_pred)  # Sensitivity
    lstm_specificity = cm_lstm[0,0] / (cm_lstm[0,0] + cm_lstm[0,1])
    lstm_f1 = f1_score(y_test, lstm_pred)
    lstm_mcc = matthews_corrcoef(y_test, lstm_pred)
    
    # ---- CNN-LSTM metrics ----
    cm_cnn = confusion_matrix(y_test, cnn_pred)
    
    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    cnn_precision = precision_score(y_test, cnn_pred)
    cnn_recall = recall_score(y_test, cnn_pred)
    cnn_specificity = cm_cnn[0,0] / (cm_cnn[0,0] + cm_cnn[0,1])
    cnn_f1 = f1_score(y_test, cnn_pred)
    cnn_mcc = matthews_corrcoef(y_test, cnn_pred)
    
    # ---- Prepare data ----
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "MCC"]
    
    lstm_values = [lstm_accuracy, lstm_precision, lstm_recall, lstm_specificity, lstm_f1, lstm_mcc]
    cnn_values  = [cnn_accuracy, cnn_precision, cnn_recall, cnn_specificity, cnn_f1, cnn_mcc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # ---- Plot ----
    plt.figure(figsize=(14,6))
    plt.bar(x - width/2, lstm_values, width, label="LSTM", color="#1f77b4")
    plt.bar(x + width/2, cnn_values, width, label="CNN-LSTM", color="#2ca02c")
    
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Performance Metrics Comparison: LSTM vs CNN-LSTM")
    
    plt.ylim(0, 1.1)
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    



.. image:: output_30_0.png

