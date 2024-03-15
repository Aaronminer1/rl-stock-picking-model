# data/data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_raw_data(file_path):
    """
    Load raw stock market data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the raw data.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded raw data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def clean_data(data):
    """
    Clean the raw stock market data.
    
    Args:
        data (pd.DataFrame): DataFrame containing the raw data.
        
    Returns:
        pd.DataFrame: DataFrame containing the cleaned data.
    """
    # Remove rows with missing values
    cleaned_data = data.dropna()
    
    # Convert date column to datetime type
    cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
    
    # Set date column as index
    cleaned_data.set_index('Date', inplace=True)
    
    return cleaned_data

def normalize_data(data):
    """
    Normalize the stock market data using Min-Max scaling.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data to be normalized.
        
    Returns:
        pd.DataFrame: DataFrame containing the normalized data.
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_data = pd.DataFrame(normalized_data, columns=data.columns, index=data.index)
    return normalized_data

def engineer_features(data):
    """
    Engineer additional features from the stock market data.
    
    Args:
        data (pd.DataFrame): DataFrame containing the stock market data.
        
    Returns:
        pd.DataFrame: DataFrame containing the data with engineered features.
    """
    # Calculate moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate relative strength index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def preprocess_data(file_path):
    """
    Preprocess the stock market data.
    
    Args:
        file_path (str): Path to the CSV file containing the raw data.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    raw_data = load_raw_data(file_path)
    
    if raw_data is not None:
        cleaned_data = clean_data(raw_data)
        normalized_data = normalize_data(cleaned_data)
        preprocessed_data = engineer_features(normalized_data)
        return preprocessed_data
    else:
        return None