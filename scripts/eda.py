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



def analyze_gdp_and_oil_prices(merged_data):
    """
    Analyzes and visualizes the correlation between GDP growth rates and Brent oil prices.

    Parameters:
    merged_data (pd.DataFrame): The merged dataframe containing GDP growth rates and Brent oil prices.
                                It should have 'Price' column for Brent oil prices and GDP growth rates for countries as columns.

    """
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")

    correlation_matrix = merged_data.corr()

    print(correlation_matrix)

    plt.figure(figsize=(14, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix between GDP Growth Rates and Brent Oil Prices')
    plt.show()

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Brent Oil Price (USD)', color='tab:blue')
    ax1.plot(merged_data.index, merged_data['Price'], label='Brent Oil Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('GDP Growth Rate (%)', color='tab:green')
    for country in merged_data.columns:
        if country != 'Price':  
            ax2.plot(merged_data.index, merged_data[country], label=f'{country} GDP Growth Rate', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('Brent Oil Prices and GDP Growth Rates Over Time')
    plt.show()




def analyze_unemployment_and_oil_consumption(merged_df):
    """
    Analyzes and visualizes the relationship between unemployment rates and oil consumption patterns.

    Parameters:
    merged_df (pd.DataFrame): The merged dataframe containing world unemployment rates and oil consumption data.
                              It should have 'World_x' for world unemployment rates and 'World_y' for oil consumption.
    """
    # Set general aesthetics for the plots
    sns.set_style("whitegrid")
   
    correlation_matrix = merged_df[['World_x', 'World_y']].corr()

  
    print("Correlation Matrix:")
    print(correlation_matrix)


    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix between World Unemployment Rates and Oil Consumption')
    plt.show()

    # Plot World Unemployment Rates and Oil Consumption over time
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('World Unemployment Rate (%)', color='tab:blue')
    ax1.plot(merged_df.index, merged_df['World_x'], label='World Unemployment Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('World Oil Consumption (Million Barrels per Day)', color='tab:green')
    ax2.plot(merged_df.index, merged_df['World_y'], label='World Oil Consumption', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('World Unemployment Rates and Oil Consumption Over Time')
    plt.show()


def analyze_exchange_rates_and_oil_prices(merged_df):
    """
    Analyzes and visualizes the relationship between exchange rates (USD) and oil prices.

    Parameters:
    merged_df (pd.DataFrame): The merged dataframe containing exchange rates (USD) and oil prices.
                              It should have 'Close' for exchange rates and 'Price' for oil prices.
    """
    
      # Set general aesthetics for the plots
    sns.set_style("whitegrid")
    correlation_matrix = merged_df[['Close', 'Price']].corr()

    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix between USD Exchange Rate (Close) and Oil Prices')
    plt.show()

    # Plot Exchange Rate and Oil Prices over time
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('USD Exchange Rate (Close)', color='tab:blue')
    ax1.plot(merged_df.index, merged_df['Close'], label='USD Exchange Rate (Close)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Oil Price (USD per Barrel)', color='tab:green')
    ax2.plot(merged_df.index, merged_df['Price'], label='Oil Price', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('USD Exchange Rate (Close) and Oil Prices Over Time')
    plt.show()






