import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
#import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


def find_missing_values(df):
    """
    Finds the number of missing values in a given dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze.
        
    Returns:
        pandas.Series: The number of missing values for each column in the dataframe.
    """
    return df.isnull().sum()

def find_duplicates(df):
    """
    Finds the number of duplicate rows in a given dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze.
        
    Returns:
        int: The number of duplicate rows in the dataframe.
    """
    return df.duplicated().sum()

def handle_missing_values(df):
    """
    Handles missing values in a given dataframe by forward-filling them.
    
    Args:
        df (pandas.DataFrame): The dataframe to handle missing values for.
        
    Returns:
        pandas.DataFrame: The updated dataframe with missing values handled.
    """
    return df.ffill()

def remove_duplicates(df):
    """
    Removes duplicate rows from a given dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to remove duplicates from.
        
    Returns:
        pandas.DataFrame: The updated dataframe with duplicate rows removed.
    """
    return df.drop_duplicates(inplace=True)

def correct_data_types(df):
    """
    Corrects the data types of the 'date' and 'purchase_time' columns in a given dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to correct data types for.
        
    Returns:
        pandas.DataFrame: The updated dataframe with corrected data types.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df



