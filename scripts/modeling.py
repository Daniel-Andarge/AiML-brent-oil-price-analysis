import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def brent_oil_arima_forecast(brent_oil_df):
    """
    Fits an ARIMA model to the Brent oil prices data, generates forecasts, and assesses the forecast accuracy.
    
    Parameters:
    brent_oil_df (pandas.DataFrame): A DataFrame containing the Brent oil prices data with a datetime index.
    
    Returns:
    None
    """

    
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    if 'Price' not in brent_oil_df.columns:
        raise ValueError("DataFrame must contain a 'Price' column")

    # Fit the ARIMA model
    p = 1  
    d = 1 
    q = 0  
    
    model = ARIMA(brent_oil_df['Price'], order=(p, d, q))
    model_fit = model.fit()
    
    # Evaluate the model
    print('AIC:', model_fit.aic)
    print('BIC:', model_fit.bic)
    
    # Perform diagnostic checks 
    residuals = model_fit.resid
    
    # Plot residuals using Matplotlib and Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=residuals.index, y=residuals.values)
    plt.title('ARIMA Model Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()
    
    # Generate forecasts
    forecast_periods = 60  
    forecasts = model_fit.get_forecast(steps=forecast_periods).predicted_mean
    
    # Get actual prices for comparison
    actual_prices = brent_oil_df['Price'].iloc[-forecast_periods:]
    if len(actual_prices) < forecast_periods:
        raise ValueError("Not enough data to compare forecasts. Check the forecast_periods or data length.")
    
    # Assess the accuracy of the forecasts
    mse = mean_squared_error(actual_prices, forecasts[:len(actual_prices)])
    mae = mean_absolute_error(actual_prices, forecasts[:len(actual_prices)])
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    
    # Plot the actual and forecasted prices using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(brent_oil_df.index, brent_oil_df['Price'], label='Actual Prices')
    plt.plot(actual_prices.index, forecasts[:len(actual_prices)], linestyle='--', color='red', label='Forecasted Prices')
    plt.title('Brent Oil Prices: Actual vs Forecasted')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_data(df, look_back=1):
    """
    Prepare data for LSTM model.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with time series data.
    look_back (int): Number of previous time steps to use as input features.
    
    Returns:
    numpy.ndarray, numpy.ndarray: Processed features and target arrays.
    """
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Price']])
    
    # Create sequences
    features, targets = [], []
    for i in range(len(scaled_data) - look_back):
        features.append(scaled_data[i:i + look_back, 0])
        targets.append(scaled_data[i + look_back, 0])
    return np.array(features), np.array(targets)

def build_lstm_model(input_shape):
    """
    Build LSTM model.
    
    Parameters:
    input_shape (tuple): Shape of input data (number of time steps, number of features).
    
    Returns:
    tensorflow.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train LSTM model.
    
    Parameters:
    model (tensorflow.keras.models.Sequential): Compiled LSTM model.
    X_train (numpy.ndarray): Input training data.
    y_train (numpy.ndarray): Target training data.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    
    Returns:
    None
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

def evaluate_lstm_model(model, X_test, y_test):
    """
    Evaluate LSTM model.
    
    Parameters:
    model (tensorflow.keras.models.Sequential): Trained LSTM model.
    X_test (numpy.ndarray): Input testing data.
    y_test (numpy.ndarray): Target testing data.
    
    Returns:
    float, float: Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae



