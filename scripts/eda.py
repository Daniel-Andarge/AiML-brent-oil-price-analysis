import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_brent_oil_prices(df):
    """
    Creates a line plot of the Brent oil prices over time.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the Brent oil prices, with the 'Date' column as the index and 'Price' column for the prices.

    Returns:
    None
    """

    # Set general aesthetics for the plots
    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df['Price'])
    plt.title('Brent Oil Prices Over Time')

    # Format the date labels on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Price (USD/barrel)')
    plt.grid(True)
    plt.show()


def analyze_brent_oil_price_distribution(df):
    """
    Analyzes the distribution of Brent oil prices using a histogram and a density plot.

    Parameters:
    brent_oil_df (pandas.DataFrame): A DataFrame containing the Brent oil price data, with the 'Price' column.

    Returns:
    None
    """

    
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    # Create a histogram
    plt.figure(figsize=(10, 6))
    df['Price'].hist(bins=30)
    plt.title('Histogram of Brent Oil Prices')
    plt.xlabel('Price (USD/barrel)')
    plt.ylabel('Frequency')
    plt.show()

    # Create a density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Price', shade=True)
    plt.title('Density Plot of Brent Oil Prices')
    plt.xlabel('Price (USD/barrel)')
    plt.ylabel('Density')
    plt.show()




def check_stationarity(brent_oil_df):
    """
    Checks if the Brent oil price time series is stationary using the Augmented Dickey-Fuller (ADF) test.
    If the series is non-stationary, determines the appropriate level of differencing required to achieve stationarity.

    Parameters:
    brent_oil_df (pandas.DataFrame): A DataFrame containing the Brent oil price data, with the 'Price' column.

    Returns:
    int: The level of differencing required to achieve stationarity (0 if the series is already stationary).
    """
    # Perform the Augmented Dickey-Fuller test
    result = adfuller(brent_oil_df['Price'])
    
    # Check if the series is stationary
    if result[1] > 0.05:
        print("The Brent oil price time series is non-stationary.")
        
        # Determine the appropriate level of differencing
        diff_level = 1
        while True:
            brent_oil_diff = brent_oil_df['Price'].diff(periods=diff_level).dropna()
            result = adfuller(brent_oil_diff)
            if result[1] <= 0.05:
                print(f"The Brent oil price time series becomes stationary after {diff_level} level(s) of differencing.")
                return diff_level
            diff_level += 1
    else:
        print("The Brent oil price time series is stationary.")
        return 0

def visualize_acf_pacf(brent_oil_df, diff_level=1):
    """
    Visualizes the autocorrelation function (ACF) and partial autocorrelation function (PACF) plots for the Brent oil price time series.

    Parameters:
    brent_oil_df (pandas.DataFrame): A DataFrame containing the Brent oil price data, with the 'Price' column.
    diff_level (int): The level of differencing required to achieve stationarity (default is 0).

    Returns:
    None
    """
        # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    # Perform differencing if required
    if diff_level > 0:
        brent_oil_diff = brent_oil_df['Price'].diff(periods=diff_level).dropna()
    else:
        brent_oil_diff = brent_oil_df['Price']
    
    # Plot the ACF
    plt.figure(figsize=(10, 6))
    plot_acf(brent_oil_diff, lags=20)
    plt.title('Autocorrelation Function (ACF) of Brent Oil Prices')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()
    
    # Plot the PACF
    plt.figure(figsize=(10, 6))
    plot_pacf(brent_oil_diff, lags=20)
    plt.title('Partial Autocorrelation Function (PACF) of Brent Oil Prices')
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.show()



