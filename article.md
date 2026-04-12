# Neural Networks for Time Series with Tensorflow Keras in Python

Neural networks excel at capturing complex temporal patterns that traditional statistical methods struggle to model.

Neural networks are good at capturing complex patterns and relationships. They can deal with nonlinear trends and interaction effects that traditional statistical methods can't handle. The trade-off is that the process is a black box - we don't know why the prediction is the way it is. We just know it works.

There are lots of ways to implement neural networks. In this project, I build three neural networks using Tensorflow (Keras): basic feedforward neural networks, recurrent neural networks (RNNs), and long short-term memory (LSTM). Exponential Smoothing and ARIMA are great for data with linear patterns, but traditional methods struggle with nonlinear relationships. Neural networks can model complex interactions between time-dependent features. Neural networks can also handle multiple inputs (multivariate data), which will improve forecasting.

There are also no assumptions of stationarity or predefined trends for Neural Networks. That means we can use them without having to run tests like Dickey-Fuller.

## Feedforward Neural Networks for Time Series

A feedforward neural network can be used to predict future values by using lagged observations as input features. While basic, it's a good starting point for time series forecasting.

    Feedforward Neural Network for Simple Forecasting

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Generate synthetic data
    np.random.seed(42)
    time = np.arange(100)
    data = 10 + 0.5 * time + np.sin(0.2 * time) + np.random.normal(scale=1.0, size=100)

    # Create lagged features
    def create_features(data, lag=3):
        X, y = [], []
        for i in range(len(data) - lag):
            X.append(data[i:i + lag])
            y.append(data[i + lag])
        return np.array(X), np.array(y)

    lag = 3
    X, y = create_features(data, lag=lag)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a simple feedforward neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(lag,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

    # Evaluate and predict
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', color='Blue')
    plt.plot(y_pred, label='Predicted', color='Red')
    plt.title(f'Feedforward Neural Network Forecast \n MAPE: {mape:.3f}')
    plt.legend()
    plt.savefig("NN_forecast.png")
    plt.show()

## Recurrent Neural Networks (RNNs)

Recurrent neural networks are designed to handle sequential data by maintaining a hidden state that captures information about previous time steps. This makes them ideal for time series.

    Basic RNN for Time Series

    from tensorflow.keras.layers import SimpleRNN

    # Build an RNN model
    model = tf.keras.Sequential([
        SimpleRNN(50, activation='relu', input_shape=(lag, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Reshape input for RNN (samples, timesteps, features)
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train the model
    model.fit(X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

    # Predict
    y_pred_rnn = model.predict(X_test_rnn)
    mape = mean_absolute_percentage_error(y_test, y_pred_rnn)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', color='Blue')
    plt.plot(y_pred_rnn, label='Predicted', color='Red')
    plt.title(f'Recurrent Neural Network Forecast \n MAPE: {mape:.3f}')
    plt.legend()
    plt.savefig("RNN_forecast.png")
    plt.show()

## Long Short-Term Memory (LSTM) Networks

LSTMs are a type of RNN that solve the vanishing gradient problem (ohh! ahh!). Basically, LSTMs remember previous values, which gives them a leg up on other NNs for complex time series with long-range patterns.

    Long Short-Term Memory (LSTM) Networks for time series

    from tensorflow.keras.layers import LSTM

    # Build an LSTM model
    model = tf.keras.Sequential([
        LSTM(50, activation='relu', input_shape=(lag, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train the model
    model.fit(X_train_rnn, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

    # Predict
    y_pred_lstm = model.predict(X_test_rnn)
    mape = mean_absolute_percentage_error(y_test, y_pred_lstm)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', color='Blue')
    plt.plot(y_pred_lstm, label='Predicted', color='Red')
    plt.title(f'LSTM Forecast. MAPE: {mape:.3f}')
    plt.legend()
    plt.savefig("LSTM_forecast.png")
    plt.show()

The graphs illustrate how neural networks can be great at matching complex time series, but with that level of mimicry comes worries about overfitting.

I built these using Tensorflow directly. There are lots of other ways to work with neural networks in time series, like N-BEATS through Darts.

## N-BEATS for Time Series Forecasting in Python

N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning model specifically designed for time series forecasting.

**Real-world data: ERCOT Load Data**

Initially, I used simulated data. But what about real data? Let's use data from ERCOT, the grid balancing authority in Texas.

Ercot Demand data: The RNN is much better than the basic Neural Network.

There is no clear benefit from the LSTM versus RNN.

Both the RNN and LSTM are extremely good at modeling this data.

## Conclusion

Neural networks are fast and can handle nonlinear patterns and multivariate inputs. Tensorflow (with Keras) is really easy to use. But Neural Networks can overfit the data and they are a \"black box,\" so the model itself is something we can decompose.
